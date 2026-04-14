//! Kubernetes Multi-Node Cluster Orchestrator for QLANG.
//!
//! Extends the TCP mode to dynamically spin up worker nodes in a Kubernetes
//! cluster, assign them roles (Trainer, Aggregator, Compressor), and coordinate
//! distributed training jobs.

use serde::{Deserialize, Serialize};

use crate::distributed::{TrainingJob, WorkerRole};

/// Configuration for the Kubernetes cluster connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sConfig {
    pub namespace: String,
    pub image: String,
    pub service_account: String,
    pub cpu_request: String,
    pub memory_request: String,
    pub gpu_limit: Option<usize>,
}

impl Default for K8sConfig {
    fn default() -> Self {
        Self {
            namespace: "qlang-system".into(),
            image: "mirkulix/qlang:latest".into(),
            service_account: "qlang-worker".into(),
            cpu_request: "2".into(),
            memory_request: "4Gi".into(),
            gpu_limit: None,
        }
    }
}

/// A generated Kubernetes manifest.
#[derive(Debug, Clone)]
pub struct K8sManifest {
    pub name: String,
    pub kind: String,
    pub yaml: String,
}

/// Generate Kubernetes manifests for a distributed training job.
pub fn generate_job_manifests(job: &TrainingJob, config: &K8sConfig) -> Vec<K8sManifest> {
    let mut manifests = Vec::new();

    // 1. Create a Headless Service for worker discovery
    let service_yaml = format!(
        r#"apiVersion: v1
kind: Service
metadata:
  name: qlang-job-{job_id}
  namespace: {namespace}
  labels:
    app: qlang
    job: {job_id}
spec:
  clusterIP: None
  selector:
    job: {job_id}
  ports:
    - name: tcp-qlms
      port: 9900
      targetPort: 9900"#,
        job_id = job.id,
        namespace = config.namespace,
    );

    manifests.push(K8sManifest {
        name: format!("qlang-job-{}-svc", job.id),
        kind: "Service".into(),
        yaml: service_yaml,
    });

    // 2. Create Pods/Deployments for each worker
    for worker in &job.workers {
        let role_str = match worker.role {
            WorkerRole::Trainer { .. } => "trainer",
            WorkerRole::Aggregator => "aggregator",
            WorkerRole::Evaluator => "evaluator",
            WorkerRole::Compressor => "compressor",
        };

        let mut resources = format!(
            r#"requests:
            cpu: "{}"
            memory: "{}""#,
            config.cpu_request, config.memory_request
        );

        if let Some(gpus) = config.gpu_limit {
            if matches!(worker.role, WorkerRole::Trainer { .. }) {
                resources.push_str(&format!(
                    r#"
          limits:
            nvidia.com/gpu: "{}""#,
                    gpus
                ));
            }
        }

        let worker_yaml = format!(
            r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: qlang-worker-{job_id}-{worker_id}
  namespace: {namespace}
  labels:
    app: qlang
    job: {job_id}
    role: {role}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qlang
      job: {job_id}
      worker: {worker_id}
  template:
    metadata:
      labels:
        app: qlang
        job: {job_id}
        worker: {worker_id}
        role: {role}
    spec:
      serviceAccountName: {service_account}
      containers:
        - name: qlang-node
          image: {image}
          command: ["qlang-cli", "agent", "--role", "{role}", "--job", "{job_id}", "--id", "{worker_id}"]
          ports:
            - containerPort: 9900
              name: tcp-qlms
          resources:
            {resources}
          env:
            - name: QLANG_COORDINATOR_URL
              value: "tcp://qlang-job-{job_id}.{namespace}.svc.cluster.local:9900""#,
            job_id = job.id,
            worker_id = worker.id,
            namespace = config.namespace,
            role = role_str,
            image = config.image,
            service_account = config.service_account,
            resources = resources,
        );

        manifests.push(K8sManifest {
            name: format!("qlang-worker-{}-{}", job.id, worker.id),
            kind: "Deployment".into(),
            yaml: worker_yaml,
        });
    }

    manifests
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{create_data_parallel_job, Hyperparams};

    #[test]
    fn test_generate_manifests() {
        let job = create_data_parallel_job("mnist-train", 2, Hyperparams::default());
        let config = K8sConfig::default();
        let manifests = generate_job_manifests(&job, &config);

        // 1 Service + (2 Trainers + 1 Aggregator + 1 Compressor) = 5 manifests
        assert_eq!(manifests.len(), 5);

        let svc = &manifests[0];
        assert_eq!(svc.kind, "Service");
        assert!(svc.yaml.contains("name: qlang-job-mnist-train"));

        let trainer0 = &manifests[1];
        assert_eq!(trainer0.kind, "Deployment");
        assert!(trainer0.yaml.contains("role: trainer"));
        assert!(trainer0.yaml.contains("qlang-cli"));
        assert!(trainer0.yaml.contains("tcp://qlang-job-mnist-train.qlang-system.svc.cluster.local:9900"));
    }
}
