import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { EmptyState, ErrorBanner, Spinner, StatCard } from '../components/ui.jsx'

const METRIC_KEYS = {
  boxLoss: 'train/box_loss',
  classLoss: 'train/cls_loss',
  dflLoss: 'train/dfl_loss',
  precision: 'metrics/precision(B)',
  recall: 'metrics/recall(B)',
  map50: 'metrics/mAP50(B)',
  map5095: 'metrics/mAP50-95(B)',
  valBoxLoss: 'val/box_loss',
  valClassLoss: 'val/cls_loss',
  valDflLoss: 'val/dfl_loss',
  learningRate0: 'lr/pg0',
  learningRate1: 'lr/pg1',
  learningRate2: 'lr/pg2',
}

function formatDuration(seconds) {
  if (seconds == null) return 'Not available'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return [days && `${days}d`, (days || hours) && `${hours}h`, `${minutes}m`]
    .filter(Boolean)
    .join(' ')
}

function formatMetric(value, digits = 3) {
  return typeof value === 'number' ? value.toFixed(digits) : 'Pending'
}

function MetricPanel({ title, metrics, emptyText }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-4 text-sm font-semibold text-slate-900">{title}</h2>
      {metrics?.length ? (
        <dl className="grid grid-cols-2 gap-x-5 gap-y-4 sm:grid-cols-3">
          {metrics.map(({ label, value, note }) => (
            <div key={label}>
              <dt className="text-xs text-slate-500">{label}</dt>
              <dd className="mt-0.5 text-lg font-semibold tabular-nums text-slate-900">{value}</dd>
              {note && <div className="text-[11px] text-slate-400">{note}</div>}
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-sm text-slate-500">{emptyText}</p>
      )}
    </section>
  )
}

function PlotCard({ title, src, alt, emptyText }) {
  return (
    <section className="card">
      <h3 className="mb-3 text-sm font-semibold text-slate-900">{title}</h3>
      {src ? (
        <a href={src} target="_blank" rel="noreferrer">
          <img className="max-h-96 w-full rounded-lg border border-slate-200 object-contain" src={src} alt={alt} />
        </a>
      ) : (
        <p className="text-sm text-slate-500">{emptyText}</p>
      )}
      <p className="mt-3 text-xs text-slate-500">
        License-plate detector evaluation artifact; this is not an end-to-end stolen/not-stolen matrix.
      </p>
    </section>
  )
}

function StatusIndicator({ running, status }) {
  const active = running
  return (
    <span className={`inline-flex items-center gap-2 text-sm font-medium ${active ? 'text-emerald-700' : 'text-slate-500'}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${active ? 'animate-pulse bg-emerald-500' : 'bg-slate-400'}`} />
      {active ? 'Training active' : status === 'completed' ? 'Training completed' : 'Training inactive'}
    </span>
  )
}

export default function TrainingMonitor() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const result = await api.trainingStatus()
        if (!cancelled) {
          setData(result)
          setError('')
        }
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    const timer = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const history = useMemo(
    () =>
      (data?.history ?? []).map((row, index) => ({
        epoch: row.epoch ?? index + 1,
        boxLoss: row[METRIC_KEYS.boxLoss],
        classLoss: row[METRIC_KEYS.classLoss],
        dflLoss: row[METRIC_KEYS.dflLoss],
        precision: row[METRIC_KEYS.precision],
        recall: row[METRIC_KEYS.recall],
        map50: row[METRIC_KEYS.map50],
        map5095: row[METRIC_KEYS.map5095],
        valBoxLoss: row[METRIC_KEYS.valBoxLoss],
        valClassLoss: row[METRIC_KEYS.valClassLoss],
        valDflLoss: row[METRIC_KEYS.valDflLoss],
        learningRate0: row[METRIC_KEYS.learningRate0],
        learningRate1: row[METRIC_KEYS.learningRate1],
        learningRate2: row[METRIC_KEYS.learningRate2],
      })),
    [data],
  )

  if (loading) return <Spinner label="Loading training status..." />

  const latest = data?.latest ?? {}
  const evaluation = data?.evaluation ?? {}
  const deployed = data?.deployed ?? {}
  const deployedReceipt = deployed.plate_install_receipt
  const detector = evaluation.detector
  const ocr = evaluation.ocr?.overall
  const performance = evaluation.performance
  const theft = evaluation.theft?.overall
  const epochLabel = data?.total_epochs
    ? `${data.completed_epochs} / ${data.total_epochs}`
    : `${data?.completed_epochs ?? 0}`

  return (
    <div>
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Model training</h1>
          <p className="text-sm text-slate-500">Live plate detector optimization metrics</p>
        </div>
        <StatusIndicator running={data?.running} status={data?.status} />
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      {!data?.run_name ? (
        <section className="card">
          <EmptyState title="No training run found" hint={data?.message} />
        </section>
      ) : (
        <>
          <section className="mb-5 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <div className="text-xs font-medium uppercase text-slate-500">Latest training run</div>
                <h2 className="mt-1 text-base font-semibold text-slate-900">{data.run_name}</h2>
                <p className="mt-1 text-sm text-slate-500">{data.message}</p>
              </div>
              <dl className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm sm:grid-cols-4">
                <div><dt className="text-xs text-slate-500">Device</dt><dd className="font-medium uppercase">{data.device ?? 'Unknown'}</dd></div>
                <div><dt className="text-xs text-slate-500">Image size</dt><dd className="font-medium">{data.image_size ?? 'Unknown'}</dd></div>
                <div><dt className="text-xs text-slate-500">Batch</dt><dd className="font-medium">{data.batch_size ?? 'Unknown'}</dd></div>
                <div><dt className="text-xs text-slate-500">Elapsed</dt><dd className="font-medium">{formatDuration(data.elapsed_seconds)}</dd></div>
              </dl>
            </div>
            <div className="mt-5">
              <div className="mb-1.5 flex justify-between text-xs text-slate-500">
                <span>Completed epochs</span>
                <span className="tabular-nums">{data.progress_percent}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full bg-blue-600 transition-all duration-500" style={{ width: `${Math.min(data.progress_percent, 100)}%` }} />
              </div>
            </div>
          </section>

          <section className="mb-5 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <div className="text-xs font-medium uppercase text-slate-500">Deployed live models</div>
              <h2 className="mt-1 text-base font-semibold text-slate-900">Current system inference configuration</h2>
            </div>
            <dl className="grid gap-4 text-sm sm:grid-cols-2 xl:grid-cols-4">
              <div>
                <dt className="text-xs text-slate-500">Vehicle/person model</dt>
                <dd className="mt-1 break-all font-medium text-slate-900">{deployed.vehicle_model ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Plate model</dt>
                <dd className="mt-1 break-all font-medium text-slate-900">{deployed.plate_model ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Plate source weights</dt>
                <dd className="mt-1 break-all font-medium text-slate-900">{deployedReceipt?.source_weights ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Installed at</dt>
                <dd className="mt-1 font-medium text-slate-900">{deployedReceipt?.installed_at ?? 'Unknown'}</dd>
              </div>
            </dl>
            {deployedReceipt ? (
              <p className="mt-4 text-xs text-slate-500">
                This deployed model may differ from the latest training run shown above.
              </p>
            ) : null}
          </section>

          <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="Epochs" value={epochLabel} sub={data.running ? 'Next epoch in progress' : 'Completed'} />
            <StatCard label="Box loss" value={formatMetric(latest[METRIC_KEYS.boxLoss])} sub="Lower is better" />
            <StatCard label="mAP@0.5" value={formatMetric(latest[METRIC_KEYS.map50])} sub="Validation detection quality" />
            <StatCard label="mAP@0.5:0.95" value={formatMetric(latest[METRIC_KEYS.map5095])} sub="Strict validation quality" />
          </div>

          {history.length === 0 ? (
            <section className="card">
              <EmptyState title="Epoch 1 is still in progress" hint="Charts and validation metrics will appear after the first epoch completes." />
            </section>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              <section className="card">
                <h2 className="mb-4 text-sm font-semibold text-slate-900">Training losses</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="boxLoss" name="Box loss" stroke="#2563eb" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="classLoss" name="Class loss" stroke="#dc2626" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="dflLoss" name="DFL loss" stroke="#ca8a04" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="card">
                <h2 className="mb-4 text-sm font-semibold text-slate-900">Validation quality</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="precision" name="Precision" stroke="#0891b2" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="recall" name="Recall" stroke="#16a34a" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="map50" name="mAP@0.5" stroke="#7c3aed" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>
          )}

          {history.length > 0 && (
            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              <section className="card">
                <h2 className="mb-4 text-sm font-semibold text-slate-900">Validation losses</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="valBoxLoss" name="Val box" stroke="#2563eb" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="valClassLoss" name="Val class" stroke="#dc2626" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="valDflLoss" name="Val DFL" stroke="#ca8a04" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="card">
                <h2 className="mb-4 text-sm font-semibold text-slate-900">Learning rates</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="learningRate0" name="Parameter group 0" stroke="#0891b2" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="learningRate1" name="Parameter group 1" stroke="#16a34a" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="learningRate2" name="Parameter group 2" stroke="#7c3aed" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>
          )}

          <h2 className="mb-3 mt-6 text-base font-semibold text-slate-900">Latest epoch metrics</h2>
          <div className="grid gap-4 lg:grid-cols-2">
            <MetricPanel
              title="Training and validation"
              emptyText="Available after epoch 1 completes."
              metrics={data.latest ? [
                { label: 'Train box loss', value: formatMetric(latest[METRIC_KEYS.boxLoss]) },
                { label: 'Train class loss', value: formatMetric(latest[METRIC_KEYS.classLoss]) },
                { label: 'Train DFL loss', value: formatMetric(latest[METRIC_KEYS.dflLoss]) },
                { label: 'Val box loss', value: formatMetric(latest[METRIC_KEYS.valBoxLoss]) },
                { label: 'Val class loss', value: formatMetric(latest[METRIC_KEYS.valClassLoss]) },
                { label: 'Val DFL loss', value: formatMetric(latest[METRIC_KEYS.valDflLoss]) },
              ] : null}
            />
            <MetricPanel
              title="Validation detection quality"
              emptyText="Available after epoch 1 completes."
              metrics={data.latest ? [
                { label: 'Precision', value: formatMetric(latest[METRIC_KEYS.precision]) },
                { label: 'Recall', value: formatMetric(latest[METRIC_KEYS.recall]) },
                { label: 'mAP@0.5', value: formatMetric(latest[METRIC_KEYS.map50]) },
                { label: 'mAP@0.5:0.95', value: formatMetric(latest[METRIC_KEYS.map5095]) },
                { label: 'Epoch time', value: latest.time != null ? `${formatMetric(latest.time, 1)} s` : 'Pending' },
                { label: 'Learning rate', value: formatMetric(latest[METRIC_KEYS.learningRate0], 6) },
              ] : null}
            />
          </div>

          <h2 className="mb-3 mt-6 text-base font-semibold text-slate-900">Final evaluation</h2>
          <div className="grid gap-4 lg:grid-cols-2">
            <MetricPanel
              title="Detector test set"
              emptyText="Detector evaluation has not been run on the selected checkpoint."
              metrics={detector ? [
                { label: 'Precision', value: formatMetric(detector.precision) },
                { label: 'Recall', value: formatMetric(detector.recall) },
                { label: 'F1 score', value: formatMetric(detector.f1_score) },
                { label: 'mAP@0.5', value: formatMetric(detector.map50) },
                { label: 'mAP@0.5:0.95', value: formatMetric(detector.map50_95) },
              ] : null}
            />
            <MetricPanel
              title="OCR test set"
              emptyText="OCR evaluation has not been run with a labelled OCR manifest."
              metrics={ocr ? [
                { label: 'Character accuracy', value: formatMetric(ocr.character_accuracy) },
                { label: 'Character error rate', value: formatMetric(ocr.character_error_rate) },
                { label: 'Exact plate accuracy', value: formatMetric(ocr.exact_plate_accuracy) },
                { label: 'Samples', value: ocr.samples ?? 'Pending' },
                { label: 'Edit errors', value: ocr.edit_errors ?? 'Pending' },
                { label: 'Characters', value: ocr.total_characters ?? 'Pending' },
              ] : null}
            />
            <MetricPanel
              title="Runtime performance"
              emptyText="Performance benchmarking has not been run."
              metrics={performance ? [
                { label: 'Throughput', value: `${formatMetric(performance.fps, 2)} FPS` },
                { label: 'Mean latency', value: `${formatMetric(performance.mean_ms, 2)} ms` },
                { label: 'Median latency', value: `${formatMetric(performance.median_ms, 2)} ms` },
                { label: 'P95 latency', value: `${formatMetric(performance.p95_ms, 2)} ms` },
                { label: 'Minimum latency', value: `${formatMetric(performance.min_ms, 2)} ms` },
                { label: 'Maximum latency', value: `${formatMetric(performance.max_ms, 2)} ms` },
                { label: 'Standard deviation', value: `${formatMetric(performance.stdev_ms, 2)} ms` },
                { label: 'Benchmark runs', value: performance.runs ?? 'Pending' },
                { label: 'Device', value: performance.device ?? 'Unknown' },
              ] : null}
            />
            <MetricPanel
              title="End-to-end theft decisions"
              emptyText="The labelled theft-scenario evaluation has not been run."
              metrics={theft ? [
                { label: 'Accuracy', value: formatMetric(theft.accuracy) },
                { label: 'Precision', value: formatMetric(theft.precision) },
                { label: 'Recall', value: formatMetric(theft.recall) },
                { label: 'F1 score', value: formatMetric(theft.f1_score) },
                { label: 'False-alert rate', value: formatMetric(theft.false_alert_rate) },
                { label: 'True positives', value: theft.true_positives ?? 'Pending' },
                { label: 'True negatives', value: theft.true_negatives ?? 'Pending' },
                { label: 'False positives', value: theft.false_positives ?? 'Pending' },
                { label: 'False negatives', value: theft.false_negatives ?? 'Pending' },
                { label: 'Mean latency', value: theft.mean_processing_ms != null ? `${formatMetric(theft.mean_processing_ms, 2)} ms` : 'Pending' },
                { label: 'P95 latency', value: theft.p95_processing_ms != null ? `${formatMetric(theft.p95_processing_ms, 2)} ms` : 'Pending' },
              ] : null}
            />
          </div>

          <h2 className="mb-3 mt-6 text-base font-semibold text-slate-900">Evaluation plots</h2>
          <section className="grid gap-4 lg:grid-cols-2">
            <PlotCard
              title="Confusion matrix"
              src={data.artifacts?.confusion_matrix}
              alt="License-plate detector confusion matrix"
              emptyText="The confusion matrix was not generated for this run."
            />
            <PlotCard
              title="Normalized confusion matrix"
              src={data.artifacts?.confusion_matrix_normalized}
              alt="Normalized license-plate detector confusion matrix"
              emptyText="The normalized confusion matrix was not generated for this run."
            />
          </section>

          <section className="mt-5 border-t border-slate-200 pt-4 text-xs text-slate-500">
            Checkpoints: last {data.checkpoints?.last_available ? 'available' : 'pending'}; best {data.checkpoints?.best_available ? 'available' : 'pending'}. Status refreshes every five seconds. Final metrics are loaded from completed evaluation reports and are not recalculated during training.
          </section>
        </>
      )}
    </div>
  )
}
