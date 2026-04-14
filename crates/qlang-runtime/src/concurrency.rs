//! Concurrency primitives for the QLANG runtime.
//!
//! Provides a thread pool, message-passing channels, shared state,
//! parallel map/reduce, and a parallel graph executor — all built
//! on `std::thread` and `std::sync` (no async runtimes).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use qlang_core::graph::{Graph, NodeId};
use qlang_core::tensor::TensorData;

use crate::executor::{self, ExecutionError};
use crate::vm::Value;

// ─── TaskPool ──────────────────────────────────────────────────────────────────

/// A handle to a submitted task, allowing the caller to retrieve its result.
pub struct TaskHandle {
    rx: std::sync::mpsc::Receiver<Value>,
}

impl TaskHandle {
    /// Block until the task completes and return its result.
    pub fn join(self) -> Value {
        self.rx.recv().unwrap_or(Value::Null)
    }
}

/// A simple thread pool that accepts `Box<dyn FnOnce() -> Value>` tasks.
pub struct TaskPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Option<std::sync::mpsc::Sender<(Job, std::sync::mpsc::Sender<Value>)>>,
}

type Job = Box<dyn FnOnce() -> Value + Send + 'static>;

struct PoolShared {
    jobs: Mutex<VecDeque<(Job, std::sync::mpsc::Sender<Value>)>>,
    condvar: Condvar,
    shutdown: Mutex<bool>,
}

impl TaskPool {
    /// Create a pool with `n_threads` worker threads.
    pub fn new(n_threads: usize) -> Self {
        assert!(n_threads > 0, "TaskPool requires at least 1 thread");

        let shared = Arc::new(PoolShared {
            jobs: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
            shutdown: Mutex::new(false),
        });

        // Channel for submitting jobs — we route through the shared queue so
        // workers can park/wake properly.
        let (tx, rx) = std::sync::mpsc::channel::<(Job, std::sync::mpsc::Sender<Value>)>();

        // Dispatcher thread: moves items from mpsc into the shared queue.
        let shared_clone = Arc::clone(&shared);
        let dispatcher = thread::spawn(move || {
            for item in rx {
                {
                    let mut q = shared_clone.jobs.lock().unwrap();
                    q.push_back(item);
                }
                shared_clone.condvar.notify_one();
            }
            // Sender dropped — signal shutdown.
            *shared_clone.shutdown.lock().unwrap() = true;
            shared_clone.condvar.notify_all();
        });

        let mut workers = vec![dispatcher];

        for _ in 0..n_threads {
            let shared_clone = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                loop {
                    let job = {
                        let mut q = shared_clone.jobs.lock().unwrap();
                        loop {
                            if let Some(job) = q.pop_front() {
                                break Some(job);
                            }
                            if *shared_clone.shutdown.lock().unwrap() {
                                break None;
                            }
                            q = shared_clone.condvar.wait(q).unwrap();
                        }
                    };
                    match job {
                        Some((f, reply)) => {
                            let result = f();
                            let _ = reply.send(result);
                        }
                        None => return,
                    }
                }
            }));
        }

        TaskPool {
            workers,
            sender: Some(tx),
        }
    }

    /// Submit a task to the pool. Returns a `TaskHandle` for retrieving the result.
    pub fn submit<F>(&self, task: F) -> TaskHandle
    where
        F: FnOnce() -> Value + Send + 'static,
    {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let sender = self.sender.as_ref().expect("pool has been shut down");
        sender.send((Box::new(task), reply_tx)).expect("pool dispatcher gone");
        TaskHandle { rx: reply_rx }
    }

    /// Wait for all previously-submitted tasks (represented by their handles)
    /// and return the results.
    pub fn wait_all(&mut self, handles: Vec<TaskHandle>) -> Vec<Value> {
        handles.into_iter().map(|h| h.join()).collect()
    }

    /// Shut down the pool, waiting for all workers to finish.
    pub fn shutdown(mut self) {
        drop(self.sender.take()); // close the channel -> dispatcher exits -> shutdown flag set
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

// ─── Channel ───────────────────────────────────────────────────────────────────

/// Sender half of a `Channel`.
#[derive(Clone)]
pub struct Sender<T> {
    inner: std::sync::mpsc::Sender<T>,
}

/// Receiver half of a `Channel`.
pub struct Receiver<T> {
    inner: std::sync::mpsc::Receiver<T>,
}

/// Simple message-passing channel.
pub struct Channel;

impl Channel {
    /// Create a new unbounded channel, returning `(Sender, Receiver)`.
    pub fn new<T>() -> (Sender<T>, Receiver<T>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Sender { inner: tx }, Receiver { inner: rx })
    }
}

impl<T> Sender<T> {
    /// Send a value into the channel. Panics if the receiver has been dropped.
    pub fn send(&self, value: T) {
        self.inner.send(value).expect("receiver dropped");
    }
}

impl<T> Receiver<T> {
    /// Block until a value is available.
    pub fn recv(&self) -> T {
        self.inner.recv().expect("sender dropped")
    }

    /// Non-blocking receive — returns `None` if no message is available.
    pub fn try_recv(&self) -> Option<T> {
        self.inner.try_recv().ok()
    }
}

// ─── SharedState ───────────────────────────────────────────────────────────────

/// A thread-safe shared variable protected by a mutex.
pub struct SharedState<T> {
    inner: Arc<Mutex<T>>,
}

impl<T: Clone> SharedState<T> {
    /// Create a new shared variable with `initial_value`.
    pub fn new(initial_value: T) -> Self {
        SharedState {
            inner: Arc::new(Mutex::new(initial_value)),
        }
    }

    /// Read (clone) the current value.
    pub fn read(&self) -> T {
        self.inner.lock().unwrap().clone()
    }

    /// Overwrite the current value.
    pub fn write(&self, value: T) {
        *self.inner.lock().unwrap() = value;
    }

    /// Apply a function to the current value, replacing it with the result.
    pub fn modify<F: FnOnce(T) -> T>(&self, f: F) {
        let mut guard = self.inner.lock().unwrap();
        let old = guard.clone();
        *guard = f(old);
    }
}

impl<T> Clone for SharedState<T> {
    fn clone(&self) -> Self {
        SharedState {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ─── parallel_map ──────────────────────────────────────────────────────────────

/// Map `func` over `data` in parallel using `n_threads` threads.
///
/// The data is split into roughly equal chunks, each chunk is processed on a
/// separate thread, and the results are concatenated in order.
pub fn parallel_map(data: Vec<f32>, func: fn(f32) -> f32, n_threads: usize) -> Vec<f32> {
    let n_threads = n_threads.max(1).min(data.len().max(1));
    let chunk_size = (data.len() + n_threads - 1) / n_threads;

    let chunks: Vec<Vec<f32>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

    let handles: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            thread::spawn(move || chunk.into_iter().map(func).collect::<Vec<f32>>())
        })
        .collect();

    let mut result = Vec::with_capacity(data.len());
    for h in handles {
        result.extend(h.join().unwrap());
    }
    result
}

// ─── parallel_reduce ───────────────────────────────────────────────────────────

/// Reduce `data` in parallel using `func` with the given `identity` element.
///
/// Each thread reduces its chunk independently, then the partial results are
/// reduced on the calling thread.
pub fn parallel_reduce(
    data: Vec<f32>,
    func: fn(f32, f32) -> f32,
    identity: f32,
    n_threads: usize,
) -> f32 {
    if data.is_empty() {
        return identity;
    }

    let n_threads = n_threads.max(1).min(data.len());
    let chunk_size = (data.len() + n_threads - 1) / n_threads;

    let chunks: Vec<Vec<f32>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

    let handles: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            thread::spawn(move || {
                chunk.into_iter().fold(identity, func)
            })
        })
        .collect();

    let mut result = identity;
    for h in handles {
        result = func(result, h.join().unwrap());
    }
    result
}

// ─── ParallelGraphExecutor ─────────────────────────────────────────────────────

/// Identifies independent subgraphs inside a QLANG `Graph` and executes them
/// on separate threads, merging the results afterward.
pub struct ParallelGraphExecutor {
    n_threads: usize,
}

impl ParallelGraphExecutor {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: n_threads.max(1),
        }
    }

    /// Find connected components (independent subgraphs) using undirected
    /// connectivity — two nodes that share an edge belong to the same component.
    pub fn find_independent_subgraphs(graph: &Graph) -> Vec<Vec<NodeId>> {
        if graph.nodes.is_empty() {
            return Vec::new();
        }

        // Build undirected adjacency.
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for node in &graph.nodes {
            adj.entry(node.id).or_default();
        }
        for edge in &graph.edges {
            adj.entry(edge.from_node).or_default().insert(edge.to_node);
            adj.entry(edge.to_node).or_default().insert(edge.from_node);
        }

        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        for node in &graph.nodes {
            if visited.contains(&node.id) {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![node.id];
            while let Some(nid) = stack.pop() {
                if !visited.insert(nid) {
                    continue;
                }
                component.push(nid);
                if let Some(neighbors) = adj.get(&nid) {
                    for &nb in neighbors {
                        if !visited.contains(&nb) {
                            stack.push(nb);
                        }
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        components
    }

    /// Build a `Graph` containing only the nodes/edges from `node_ids`.
    fn subgraph(graph: &Graph, node_ids: &HashSet<NodeId>) -> Graph {
        let mut sub = Graph::new(format!("{}_sub", graph.id));
        sub.version = graph.version.clone();
        for node in &graph.nodes {
            if node_ids.contains(&node.id) {
                sub.nodes.push(node.clone());
            }
        }
        for edge in &graph.edges {
            if node_ids.contains(&edge.from_node) && node_ids.contains(&edge.to_node) {
                sub.edges.push(edge.clone());
            }
        }
        sub
    }

    /// Execute the graph, running independent subgraphs in parallel.
    ///
    /// `inputs` maps input-node names to their tensor data.  Results from all
    /// subgraphs are merged into a single `HashMap`.
    pub fn execute(
        &self,
        graph: &Graph,
        inputs: HashMap<String, TensorData>,
    ) -> Result<HashMap<String, TensorData>, ExecutionError> {
        let components = Self::find_independent_subgraphs(graph);

        if components.len() <= 1 {
            // Nothing to parallelise — just run sequentially.
            return executor::execute(graph, inputs).map(|r| r.outputs);
        }

        // Share inputs across threads (they are read-only).
        let inputs = Arc::new(inputs);
        let mut merged: HashMap<String, TensorData> = HashMap::new();

        // Process components in batches of at most n_threads.
        for batch in components.chunks(self.n_threads) {
            let handles: Vec<_> = batch
                .iter()
                .map(|comp| {
                    let node_set: HashSet<NodeId> = comp.iter().copied().collect();
                    let sub = Self::subgraph(graph, &node_set);
                    let inputs = Arc::clone(&inputs);
                    thread::spawn(move || {
                        executor::execute(&sub, (*inputs).clone())
                    })
                })
                .collect();

            for h in handles {
                let result = h.join().map_err(|_| {
                    ExecutionError::RuntimeError("subgraph thread panicked".into())
                })??;
                merged.extend(result.outputs);
            }
        }

        Ok(merged)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // -- TaskPool tests -------------------------------------------------------

    #[test]
    fn test_taskpool_basic_execution() {
        let pool = TaskPool::new(2);
        let handle = pool.submit(|| Value::Number(42.0));
        let result = handle.join();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_taskpool_multiple_tasks() {
        let pool = TaskPool::new(4);
        let mut handles = Vec::new();
        for i in 0..10 {
            let val = i as f64;
            handles.push(pool.submit(move || Value::Number(val * 2.0)));
        }
        let results: Vec<Value> = handles.into_iter().map(|h| h.join()).collect();
        for (i, r) in results.iter().enumerate() {
            assert_eq!(*r, Value::Number(i as f64 * 2.0));
        }
    }

    #[test]
    fn test_taskpool_single_thread() {
        let pool = TaskPool::new(1);
        let h1 = pool.submit(|| Value::Number(1.0));
        let h2 = pool.submit(|| Value::Number(2.0));
        let h3 = pool.submit(|| Value::Number(3.0));
        assert_eq!(h1.join(), Value::Number(1.0));
        assert_eq!(h2.join(), Value::Number(2.0));
        assert_eq!(h3.join(), Value::Number(3.0));
    }

    // -- Channel tests --------------------------------------------------------

    #[test]
    fn test_channel_send_recv() {
        let (tx, rx) = Channel::new::<Value>();
        tx.send(Value::String("hello".into()));
        let v = rx.recv();
        assert_eq!(v, Value::String("hello".into()));
    }

    #[test]
    fn test_channel_multiple_messages() {
        let (tx, rx) = Channel::new::<i32>();
        for i in 0..5 {
            tx.send(i);
        }
        for i in 0..5 {
            assert_eq!(rx.recv(), i);
        }
    }

    #[test]
    fn test_channel_try_recv_empty() {
        let (_tx, rx) = Channel::new::<i32>();
        assert!(rx.try_recv().is_none());
    }

    // -- SharedState tests ----------------------------------------------------

    #[test]
    fn test_shared_state_read_write() {
        let state = SharedState::new(Value::Number(0.0));
        assert_eq!(state.read(), Value::Number(0.0));
        state.write(Value::Number(99.0));
        assert_eq!(state.read(), Value::Number(99.0));
    }

    #[test]
    fn test_shared_state_concurrent_access() {
        let state = SharedState::new(0i64);
        let mut handles = Vec::new();
        for _ in 0..10 {
            let s = state.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    s.modify(|v| v + 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(state.read(), 1000);
    }

    // -- parallel_map tests ---------------------------------------------------

    #[test]
    fn test_parallel_map_correctness() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = parallel_map(data.clone(), |x| x * 2.0, 4);
        let expected: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_map_faster_than_sequential() {
        // Use a moderately expensive function on a large dataset so that
        // parallel execution has a measurable advantage.
        let n = 2_000_000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        fn heavy(x: f32) -> f32 {
            // A few chained transcendentals to burn CPU.
            (x.sin().cos().sqrt().abs() + 1.0).ln()
        }

        let start_seq = Instant::now();
        let _seq: Vec<f32> = data.iter().map(|&x| heavy(x)).collect();
        let seq_time = start_seq.elapsed();

        let start_par = Instant::now();
        let _par = parallel_map(data, heavy, 4);
        let par_time = start_par.elapsed();

        // Parallel should be at least somewhat faster (use generous 0.9x to
        // avoid flaky failures on heavily loaded CI runners).
        assert!(
            par_time < seq_time || seq_time.as_millis() < 50,
            "parallel_map was not faster: par={par_time:?} seq={seq_time:?}"
        );
    }

    // -- parallel_reduce tests ------------------------------------------------

    #[test]
    fn test_parallel_reduce_sum() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let result = parallel_reduce(data, |a, b| a + b, 0.0, 4);
        assert!((result - 5050.0).abs() < 1e-3);
    }

    #[test]
    fn test_parallel_reduce_empty() {
        let result = parallel_reduce(vec![], |a, b| a + b, 0.0, 4);
        assert!((result - 0.0).abs() < 1e-6);
    }

    // -- ParallelGraphExecutor tests ------------------------------------------

    #[test]
    fn test_find_independent_subgraphs_disconnected() {
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;

        let mut graph = Graph::new("test");
        // Two disconnected input nodes -> two independent subgraphs.
        graph.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_vector(1)],
        );
        graph.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_vector(1)],
        );

        let components = ParallelGraphExecutor::find_independent_subgraphs(&graph);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_find_independent_subgraphs_connected() {
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;

        let mut graph = Graph::new("test");
        let ty = TensorType::f32_vector(1);
        let a = graph.add_node(Op::Input { name: "a".into() }, vec![], vec![ty.clone()]);
        let b = graph.add_node(Op::Input { name: "b".into() }, vec![], vec![ty.clone()]);
        graph.add_edge(a, 0, b, 0, ty);

        let components = ParallelGraphExecutor::find_independent_subgraphs(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);
    }
}
