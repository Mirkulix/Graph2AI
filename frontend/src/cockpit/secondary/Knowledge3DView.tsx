import { lazy, Suspense } from 'react';
import { SecondaryFrame, Empty } from './FederationView';

const KnowledgeGraph3DView = lazy(() => import('../../KnowledgeGraph3DView'));

export default function Knowledge3DView() {
  return (
    <SecondaryFrame title="knowledge 3D" subtitle="three.js force-directed visualization of the graph corpus">
      <Suspense fallback={<Empty text="Loading 3D engine…" />}>
        <div style={{ height: 540, background: 'var(--bg-panel)', border: '1px solid var(--rule-default)', borderRadius: 4, overflow: 'hidden' }}>
          <KnowledgeGraph3DView />
        </div>
      </Suspense>
    </SecondaryFrame>
  );
}
