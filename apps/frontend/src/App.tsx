import { useState, useEffect } from 'react'
import './App.css'
import { StartAnalysis, SetOracleModifier, GetStats, GetResults } from '../wailsjs/go/main/App'
import { EventsOn } from '../wailsjs/runtime/runtime'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, BarChart, Bar } from 'recharts'
import { Home, BarChart3, FlaskConical, FileText, Activity, Zap } from 'lucide-react'

interface CustomerResult {
  customer_id: string
  tenure: number
  monthly_charges: number
  churn_probability: number
  risk_level: string
  action_name: string
  reasoning: string
}

interface Stats {
  total_customers: number
  high_risk: number
  medium_risk: number
  low_risk: number
  est_revenue_saved: number
  processing_time_ms: number
}

const COLORS = ['#ef4444', '#f59e0b', '#22c55e']

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home, desc: 'Real-time monitoring' },
  { id: 'analytics', label: 'Analytics', icon: BarChart3, desc: 'Business intelligence' },
  { id: 'training', label: 'Training Lab', icon: FlaskConical, desc: 'Model management' },
  { id: 'reports', label: 'Reports', icon: FileText, desc: 'Export & filtering' },
]

interface DatasetInfo {
  name: string
  trained: boolean
  created_at: string
}

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [logs, setLogs] = useState<CustomerResult[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [isRunning, setIsRunning] = useState(false)
  const [brainReady, setBrainReady] = useState(false)
  const [oracle, setOracle] = useState(1.0)
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [activeDataset, setActiveDataset] = useState('Default Dataset')

  useEffect(() => {
    EventsOn('brain-ready', () => setBrainReady(true))
    EventsOn('analysis-start', (total: number) => {
      setIsRunning(true)
      setProgress({ current: 0, total })
      setLogs([])
    })
    EventsOn('analysis-progress', (data: { current: number; total: number; customer: CustomerResult }) => {
      setProgress({ current: data.current, total: data.total })
      setLogs(prev => [data.customer, ...prev].slice(0, 25))
    })
    EventsOn('analysis-complete', (s: Stats) => {
      setStats(s)
      setIsRunning(false)
    })
    EventsOn('oracle-updated', (val: number) => setOracle(val))
    EventsOn('datasets-updated', (ds: DatasetInfo[]) => setDatasets(ds))
    EventsOn('dataset-changed', (name: string) => setActiveDataset(name))
    
    GetStats().then(s => { if (s.total_customers > 0) setStats(s) })
    
    const loadDatasets = async () => {
      try {
        const module = await import('../wailsjs/go/main/App') as any
        if (module.GetAvailableDatasets) {
          const ds = await module.GetAvailableDatasets()
          if (ds) setDatasets(ds)
        }
        if (module.GetActiveDataset) {
          const active = await module.GetActiveDataset()
          if (active) setActiveDataset(active)
        }
      } catch {}
    }
    loadDatasets()
    
    const checkBrain = async () => {
      try {
        const res = await fetch('http://localhost:5000/health')
        if (res.ok) setBrainReady(true)
      } catch {}
    }
    checkBrain()
    const interval = setInterval(checkBrain, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleStart = () => StartAnalysis()
  const handleOracleChange = (val: number) => {
    setOracle(val)
    SetOracleModifier(val)
  }
  
  const handleDatasetChange = async (name: string) => {
    try {
      const module = await import('../wailsjs/go/main/App') as any
      if (module.SetActiveDataset) {
        await module.SetActiveDataset(name)
        setActiveDataset(name)
      }
    } catch (e) {
      console.error('Failed to set dataset:', e)
    }
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">🛡️</div>
            <div>
              <h1 className="logo-text">AURA-SENTINEL</h1>
              <span className="logo-sub">Enterprise AI Platform</span>
            </div>
          </div>
        </div>
        <nav className="nav-menu">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
            >
              <item.icon size={18} />
              <div className="nav-text">
                <span className="nav-label">{item.label}</span>
                <span className="nav-desc">{item.desc}</span>
              </div>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="system-status">
            <div className="status-header">
              <div className={`status-dot ${brainReady ? 'active' : ''}`}></div>
              <span>System Status</span>
            </div>
            <div className="status-info">
              <p>Engine: <span className="text-green">Online</span></p>
              <p>Brain API: <span className={brainReady ? 'text-green' : 'text-yellow'}>
                {brainReady ? 'Connected' : 'Starting...'}
              </span></p>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {currentPage === 'dashboard' && (
          <DashboardPage
            logs={logs}
            stats={stats}
            progress={progress}
            isRunning={isRunning}
            brainReady={brainReady}
            oracle={oracle}
            onStart={handleStart}
            onOracleChange={handleOracleChange}
            datasets={datasets}
            activeDataset={activeDataset}
            onDatasetChange={handleDatasetChange}
          />
        )}
        {currentPage === 'analytics' && <AnalyticsPage stats={stats} activeDataset={activeDataset} />}
        {currentPage === 'training' && <TrainingPage />}
        {currentPage === 'reports' && <ReportsPage stats={stats} logs={logs} datasets={datasets} activeDataset={activeDataset} onDatasetChange={handleDatasetChange} />}
      </main>
    </div>
  )
}

function DashboardPage({ logs, stats, progress, isRunning, brainReady, oracle, onStart, onOracleChange, datasets, activeDataset, onDatasetChange }: any) {
  const progressPercent = progress.total > 0 ? (progress.current / progress.total) * 100 : 0
  const pieData = stats ? [
    { name: 'High', value: stats.high_risk },
    { name: 'Medium', value: stats.medium_risk },
    { name: 'Low', value: stats.low_risk },
  ] : []
  const chartData = [
    { ep: '100', rl: 45, base: 50 },
    { ep: '300', rl: 62, base: 52 },
    { ep: '500', rl: 78, base: 49 },
    { ep: '700', rl: 85, base: 51 },
    { ep: '1000', rl: 92, base: 50 },
  ]

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Command Center</h1>
          <p className="page-subtitle">Real-time AI-powered customer retention monitoring</p>
        </div>
        <div className="header-actions">
          <select className="dataset-select" value={activeDataset || ''} onChange={(e) => onDatasetChange?.(e.target.value)}>
            {datasets?.map((ds: any) => (
              <option key={ds.name} value={ds.name}>📊 {ds.name}</option>
            ))}
          </select>
          <div className={`ws-badge ${brainReady ? 'connected' : ''}`}>
            <Zap size={14} />
            {brainReady ? 'Brain: Ready' : 'Brain: Starting...'}
          </div>
          {oracle > 1 && <span className="oracle-badge">🔮 ORACLE ACTIVE</span>}
          {!isRunning && brainReady && (
            <button className="btn-primary" onClick={onStart}>▶ START ANALYSIS</button>
          )}
        </div>
      </header>

      <div className="oracle-panel">
        <div className="oracle-header">
          <span>🔮 ORACLE SCENARIO CONTROL <span className="oracle-hint">What-If Analysis Engine</span></span>
          <span className="oracle-value">{oracle.toFixed(1)}x Cost Modifier</span>
        </div>
        <input type="range" min="0.5" max="3.0" step="0.1" value={oracle} onChange={(e) => onOracleChange(parseFloat(e.target.value))} className="oracle-slider" />
        <div className="oracle-labels">
          <span>💰 Cheaper Discounts</span>
          <span>⚖️ Normal</span>
          <span>📈 Expensive Discounts</span>
        </div>
      </div>

      {isRunning && (
        <div className="progress-panel">
          <div className="progress-text">
            <span>Processing customers...</span>
            <span>{progress.current.toLocaleString()} / {progress.total.toLocaleString()}</span>
          </div>
          <div className="progress-bar"><div className="progress-fill" style={{ width: `${progressPercent}%` }}></div></div>
        </div>
      )}

      <div className="dashboard-grid">
        <div className="feed-panel">
          <h2 className="panel-title"><Activity size={16} /> LIVE MATRIX FEED</h2>
          <div className="feed-list">
            {logs.map((log: CustomerResult, i: number) => (
              <div key={i} className={`feed-item ${log.risk_level.toLowerCase()}`}>
                <div className="feed-header">
                  <span className="customer-id">{log.customer_id}</span>
                  <span className="risk-badge">{log.risk_level}</span>
                </div>
                <div className="feed-body">
                  <span>Prob: {(log.churn_probability * 100).toFixed(0)}%</span>
                  <span>→ {log.action_name}</span>
                </div>
                {log.reasoning && <div className="feed-reason">💡 {log.reasoning}</div>}
              </div>
            ))}
            {logs.length === 0 && <div className="empty-state">🎯<p>{brainReady ? 'Click START ANALYSIS to begin' : 'Waiting for Brain API...'}</p></div>}
          </div>
        </div>

        <div className="stats-area">
          {stats && (
            <div className="stats-grid">
              <StatCard icon="👥" label="Analyzed" value={stats.total_customers.toLocaleString()} />
              <StatCard icon="🔴" label="High Risk" value={stats.high_risk.toLocaleString()} accent="red" />
              <StatCard icon="🟡" label="Medium" value={stats.medium_risk.toLocaleString()} accent="yellow" />
              <StatCard icon="💰" label="Saved" value={`$${(stats.est_revenue_saved/1000).toFixed(0)}K`} accent="purple" />
            </div>
          )}

          <div className="chart-panel">
            <h3 className="panel-title">📈 RL Agent Performance</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="ep" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="rl" stroke="#10b981" strokeWidth={3} name="RL Agent" dot={{ fill: '#10b981', r: 4 }} />
                  <Line type="monotone" dataKey="base" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" name="Baseline" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bottom-panels">
            <div className="chart-panel">
              <h3 className="panel-title">🎯 Risk Distribution</h3>
              <div className="chart-container small">
                {pieData.length > 0 && pieData.some(d => d.value > 0) ? (
                  <ResponsiveContainer width="100%" height={140}>
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" innerRadius={35} outerRadius={55} dataKey="value" label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`} labelLine={false}>
                        {pieData.map((_, index) => (<Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                ) : <div className="empty-state small">Awaiting data...</div>}
              </div>
            </div>

            <div className="chart-panel">
              <h3 className="panel-title">⚡ System Metrics</h3>
              <div className="metrics-list">
                <div className="metric-row"><span>Oracle Mode</span><span className={oracle > 1 ? 'text-purple' : ''}>{oracle > 1 ? `Active (${oracle.toFixed(1)}x)` : 'Disabled'}</span></div>
                <div className="metric-row"><span>Brain API</span><span className="text-green">Connected</span></div>
                <div className="metric-row"><span>Processing</span><span>{stats ? `${stats.processing_time_ms}ms` : '—'}</span></div>
                <div className="metric-row"><span>Model</span><span className="text-green">XGBoost + DQN</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function AnalyticsPage({ stats, activeDataset }: any) {
  const trendData = [
    { month: 'Jul', churn: 23, retained: 77 },
    { month: 'Aug', churn: 21, retained: 79 },
    { month: 'Sep', churn: 25, retained: 75 },
    { month: 'Oct', churn: 19, retained: 81 },
    { month: 'Nov', churn: 17, retained: 83 },
    { month: 'Dec', churn: 15, retained: 85 },
    { month: 'Jan', churn: 12, retained: 88 },
  ]
  const radarData = [
    { metric: 'Accuracy', value: 94 },
    { metric: 'Precision', value: 91 },
    { metric: 'Recall', value: 88 },
    { metric: 'F1 Score', value: 89 },
    { metric: 'AUC-ROC', value: 96 },
  ]
  const cohortData = [
    { tenure: '0-6 mo', churn: 42 },
    { tenure: '6-12 mo', churn: 28 },
    { tenure: '12-24 mo', churn: 18 },
    { tenure: '24-36 mo', churn: 12 },
    { tenure: '36+ mo', churn: 8 },
  ]

  return (
    <div className="page">
      <header className="page-header">
        <div><h1 className="page-title">Analytics Dashboard</h1><p className="page-subtitle">Historical trends and aggregate insights</p></div>
      </header>
      <div className="stats-grid four">
        <StatCard icon="👥" label="Total Analyzed" value={stats?.total_customers?.toLocaleString() || '—'} />
        <StatCard icon="🔴" label="High Risk" value={stats?.high_risk?.toLocaleString() || '—'} accent="red" />
        <StatCard icon="🟡" label="Medium Risk" value={stats?.medium_risk?.toLocaleString() || '—'} accent="yellow" />
        <StatCard icon="🟢" label="Low Risk" value={stats?.low_risk?.toLocaleString() || '—'} accent="green" />
      </div>
      <div className="analytics-grid">
        <div className="chart-panel wide">
          <h3 className="panel-title">📈 Retention Trend (Monthly)</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="month" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '11px' }} />
                <Area type="monotone" dataKey="retained" stroke="#10b981" fill="#10b981" fillOpacity={0.3} name="Retained %" />
                <Area type="monotone" dataKey="churn" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} name="Churned %" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="chart-panel">
          <h3 className="panel-title">🧠 AI Model Performance</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={radarData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" fontSize={10} domain={[0, 100]} />
                <YAxis dataKey="metric" type="category" stroke="#64748b" fontSize={9} width={70} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '11px' }} />
                <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} name="Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="chart-panel">
          <h3 className="panel-title">📊 Churn by Tenure Cohort</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={cohortData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" fontSize={10} />
                <YAxis dataKey="tenure" type="category" stroke="#64748b" fontSize={9} width={60} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '11px' }} />
                <Bar dataKey="churn" fill="#f59e0b" radius={[0, 4, 4, 0]} name="Churn %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

interface TrainingRecord {
  id: number
  filename: string
  status: string
  accuracy: number
  created_at: string
}

function TrainingPage() {
  const [history, setHistory] = useState<TrainingRecord[]>([])
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const { GetTrainingHistory } = await import('../wailsjs/go/main/App')
        const h = await GetTrainingHistory()
        if (h) setHistory(h)
      } catch {}
    }
    loadHistory()
    
    EventsOn('training-updated', (h: TrainingRecord[]) => setHistory(h))
  }, [])

  const handleSelectFile = async () => {
    setUploading(true)
    try {
      const { SelectDataset, UploadDataset } = await import('../wailsjs/go/main/App')
      const filePath = await SelectDataset()
      if (filePath) {
        await UploadDataset(filePath)
      }
    } catch (e) {
      console.error('Upload error:', e)
    }
    setUploading(false)
  }

  const handleTrain = async (id: number) => {
    try {
      const { TriggerTraining } = await import('../wailsjs/go/main/App')
      await TriggerTraining(id)
    } catch (e) {
      console.error('Training error:', e)
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div><h1 className="page-title">Training Lab</h1><p className="page-subtitle">Upload datasets and train custom AI models</p></div>
      </header>
      <div className="training-grid">
        <div className="chart-panel">
          <h3 className="panel-title">📤 Upload Dataset</h3>
          <div className="upload-zone" onClick={handleSelectFile}>
            <div className="upload-icon">{uploading ? '⏳' : '📁'}</div>
            <p>{uploading ? 'Selecting file...' : 'Click to select file'}</p>
            <span>Supports: CSV, XLSX (max 10MB)</span>
          </div>
          <button className="btn-primary full" onClick={handleSelectFile} disabled={uploading}>
            {uploading ? '⏳ Selecting...' : '⬆️ Upload Dataset'}
          </button>
        </div>
        <div className="chart-panel">
          <h3 className="panel-title">📜 Training History</h3>
          {history.length === 0 ? (
            <div className="empty-state">🧪<p>No training records yet</p><span>Upload a dataset to start</span></div>
          ) : (
            <div className="training-list">
              {history.map(record => (
                <div key={record.id} className={`training-item ${record.status.toLowerCase()}`}>
                  <div className="training-info">
                    <span className="training-filename">📄 {record.filename}</span>
                    <span className="training-date">{record.created_at}</span>
                  </div>
                  <div className="training-actions">
                    <span className={`training-status ${record.status.toLowerCase()}`}>{record.status}</span>
                    {record.accuracy > 0 && <span className="training-acc">{record.accuracy.toFixed(1)}%</span>}
                    {record.status === 'PENDING' && (
                      <button className="btn-train" onClick={() => handleTrain(record.id)}>🚀 Train</button>
                    )}
                    {record.status === 'TRAINING' && <span className="training-spinner">⏳</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ReportsPage({ stats, logs, datasets, activeDataset, onDatasetChange }: any) {
  const [filter, setFilter] = useState('ALL')
  const [exporting, setExporting] = useState(false)
  const filters = ['ALL', 'HIGH', 'MEDIUM', 'LOW']
  
  const filteredLogs = logs.filter((l: any) => filter === 'ALL' || l.risk_level === filter)

  const exportCSV = () => {
    const headers = ['CustomerID', 'Tenure', 'MonthlyCharges', 'ChurnProbability', 'RiskLevel', 'Action']
    const rows = filteredLogs.map((c: any) => [
      c.customer_id, c.tenure, c.monthly_charges?.toFixed(2) || '0',
      (c.churn_probability * 100).toFixed(2) + '%', c.risk_level, c.action_name
    ])
    const csv = [headers.join(','), ...rows.map((r: any) => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `aura-sentinel-report-${filter.toLowerCase()}-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
  }

  const exportPDF = async () => {
    setExporting(true)
    const { default: jsPDF } = await import('jspdf')
    const doc = new jsPDF()
    const pageWidth = doc.internal.pageSize.getWidth()
    
    // Header
    doc.setFillColor(10, 14, 26)
    doc.rect(0, 0, pageWidth, 45, 'F')
    doc.setFontSize(20)
    doc.setTextColor(16, 185, 129)
    doc.text('AURA-SENTINEL', 20, 22)
    doc.setFontSize(10)
    doc.setTextColor(100)
    doc.text('Customer Retention Analysis Report', 20, 32)
    doc.text(`Generated: ${new Date().toLocaleString()} | Filter: ${filter}`, 20, 40)
    
    // Stats
    let y = 55
    doc.setFontSize(12)
    doc.setTextColor(0)
    doc.text(`Total: ${stats?.total_customers || 0} | High: ${stats?.high_risk || 0} | Medium: ${stats?.medium_risk || 0} | Low: ${stats?.low_risk || 0}`, 20, y)
    
    // Table header
    y = 70
    doc.setFillColor(30, 41, 59)
    doc.rect(15, y - 5, pageWidth - 30, 10, 'F')
    doc.setFontSize(9)
    doc.setTextColor(16, 185, 129)
    doc.text('Customer ID', 20, y + 2)
    doc.text('Tenure', 60, y + 2)
    doc.text('Monthly $', 85, y + 2)
    doc.text('Churn %', 115, y + 2)
    doc.text('Risk', 140, y + 2)
    doc.text('Action', 160, y + 2)
    
    // Table data
    y += 12
    doc.setTextColor(60)
    filteredLogs.slice(0, 40).forEach((c: any, i: number) => {
      if (y > 270) return
      doc.setFontSize(8)
      doc.text(c.customer_id || '', 20, y)
      doc.text(`${c.tenure} mo`, 60, y)
      doc.text(`$${c.monthly_charges?.toFixed(0) || '0'}`, 85, y)
      doc.text(`${(c.churn_probability * 100).toFixed(1)}%`, 115, y)
      doc.text(c.risk_level || '', 140, y)
      doc.text(c.action_name || '', 160, y)
      y += 6
    })
    
    doc.save(`aura-sentinel-report-${filter.toLowerCase()}-${new Date().toISOString().split('T')[0]}.pdf`)
    setExporting(false)
  }

  return (
    <div className="page">
      <header className="page-header">
        <div><h1 className="page-title">Reports</h1><p className="page-subtitle">Filter, analyze, and export customer data</p></div>
        <div className="header-actions">
          <button className="btn-secondary" onClick={exportCSV} disabled={logs.length === 0}>📄 Export CSV</button>
          <button className="btn-primary" onClick={exportPDF} disabled={logs.length === 0 || exporting}>
            {exporting ? '⏳ Generating...' : '📑 Export PDF'}
          </button>
        </div>
      </header>
      <div className="filter-bar">
        {filters.map(f => (
          <button key={f} onClick={() => setFilter(f)} className={`filter-btn ${filter === f ? 'active' : ''} ${f.toLowerCase()}`}>{f === 'ALL' ? '👥 All' : f === 'HIGH' ? '🔴 High' : f === 'MEDIUM' ? '🟡 Medium' : '🟢 Low'}</button>
        ))}
      </div>
      <div className="stats-grid four">
        <StatCard icon="📊" label="Showing" value={filteredLogs.length.toLocaleString()} />
        <StatCard icon="🔴" label="High Risk" value={filteredLogs.filter((l: any) => l.risk_level === 'HIGH').length.toLocaleString()} accent="red" />
        <StatCard icon="🟡" label="Medium" value={filteredLogs.filter((l: any) => l.risk_level === 'MEDIUM').length.toLocaleString()} accent="yellow" />
        <StatCard icon="🟢" label="Low Risk" value={filteredLogs.filter((l: any) => l.risk_level === 'LOW').length.toLocaleString()} accent="green" />
      </div>
      <div className="chart-panel">
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr><th>Customer ID</th><th>Tenure</th><th>Monthly</th><th>Churn %</th><th>Risk</th><th>Action</th></tr>
            </thead>
            <tbody>
              {filteredLogs.slice(0, 50).map((c: any, i: number) => (
                <tr key={i}>
                  <td>{c.customer_id}</td>
                  <td>{c.tenure} mo</td>
                  <td>${c.monthly_charges?.toFixed(2) || '0.00'}</td>
                  <td>{(c.churn_probability * 100).toFixed(1)}%</td>
                  <td><span className={`risk-tag ${c.risk_level.toLowerCase()}`}>{c.risk_level}</span></td>
                  <td className="text-purple">{c.action_name}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {logs.length === 0 && <div className="empty-state table">Run analysis to see data</div>}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, accent }: { icon: string; label: string; value: string; accent?: string }) {
  return (
    <div className={`stat-card ${accent || ''}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  )
}

export default App
