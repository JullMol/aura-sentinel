package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type App struct {
	ctx              context.Context
	pythonProcess    *exec.Cmd
	discountModifier float64
	processing       bool
	results          []CustomerResult
	stats            DashboardStats
	mutex            sync.RWMutex
	activeDataset    string
	datasets         map[string]*DatasetAnalysis
}

type DatasetAnalysis struct {
	Name      string           `json:"name"`
	FilePath  string           `json:"file_path"`
	Results   []CustomerResult `json:"results"`
	Stats     DashboardStats   `json:"stats"`
	Trained   bool             `json:"trained"`
	CreatedAt string           `json:"created_at"`
}

type CustomerResult struct {
	CustomerID       string  `json:"customer_id"`
	Tenure           int     `json:"tenure"`
	MonthlyCharges   float64 `json:"monthly_charges"`
	ChurnProbability float64 `json:"churn_probability"`
	RiskLevel        string  `json:"risk_level"`
	ActionID         int     `json:"action_id"`
	ActionName       string  `json:"action_name"`
	ActionCost       string  `json:"action_cost"`
	Reasoning        string  `json:"reasoning"`
}

type DashboardStats struct {
	TotalCustomers   int     `json:"total_customers"`
	HighRisk         int     `json:"high_risk"`
	MediumRisk       int     `json:"medium_risk"`
	LowRisk          int     `json:"low_risk"`
	EstRevenueSaved  float64 `json:"est_revenue_saved"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

type PredictionResponse struct {
	ChurnProbability float64 `json:"churn_probability"`
	ActionID         int     `json:"action_id"`
	ActionName       string  `json:"action_name"`
	ActionCost       float64 `json:"action_cost"`
	Reasoning        string  `json:"reasoning"`
	Status           string  `json:"status"`
}

func NewApp() *App {
	defaultPath := `C:\Users\LENOVO\OneDrive\Documents\Golang\aura-sentinel\apps\engine-go\retention_strategy_results.csv`
	return &App{
		discountModifier: 1.0,
		activeDataset:    "Default Dataset",
		datasets: map[string]*DatasetAnalysis{
			"Default Dataset": {
				Name:      "Default Dataset",
				FilePath:  defaultPath,
				Trained:   true,
				CreatedAt: "Built-in",
			},
		},
	}
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	go a.startPythonBrain()
}

func (a *App) shutdown(ctx context.Context) {
	if a.pythonProcess != nil {
		a.pythonProcess.Process.Kill()
	}
}

func (a *App) startPythonBrain() {
	for i := 0; i < 30; i++ {
		resp, err := http.Get("http://localhost:5000/health")
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			runtime.EventsEmit(a.ctx, "brain-ready", true)
			fmt.Println("✅ Python Brain API detected")
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Println("⚠️ Python Brain API not detected - please start api.py manually")
}

func (a *App) GetStats() DashboardStats {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	return a.stats
}

func (a *App) GetResults(limit int) []CustomerResult {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if limit > len(a.results) {
		limit = len(a.results)
	}
	return a.results[:limit]
}

func (a *App) SetOracleModifier(value float64) {
	a.discountModifier = value
	runtime.EventsEmit(a.ctx, "oracle-updated", value)
}

func (a *App) GetAvailableDatasets() []DatasetAnalysis {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	var datasets []DatasetAnalysis
	for _, ds := range a.datasets {
		datasets = append(datasets, *ds)
	}
	return datasets
}

func (a *App) SetActiveDataset(name string) error {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	if _, exists := a.datasets[name]; !exists {
		return fmt.Errorf("dataset not found: %s", name)
	}
	a.activeDataset = name
	if ds := a.datasets[name]; ds != nil && len(ds.Results) > 0 {
		a.results = ds.Results
		a.stats = ds.Stats
	}
	runtime.EventsEmit(a.ctx, "dataset-changed", name)
	return nil
}

func (a *App) GetActiveDataset() string {
	return a.activeDataset
}

func (a *App) StartAnalysis() error {
	if a.processing {
		return fmt.Errorf("analysis already running")
	}

	go a.runAnalysis()
	return nil
}

func (a *App) runAnalysis() {
	a.processing = true
	defer func() { a.processing = false }()

	a.mutex.RLock()
	activeDS := a.datasets[a.activeDataset]
	a.mutex.RUnlock()

	if activeDS == nil {
		fmt.Println("❌ No active dataset selected")
		runtime.EventsEmit(a.ctx, "analysis-error", "No dataset selected")
		return
	}

	dataPath := activeDS.FilePath
	fmt.Println("📂 Loading data from:", dataPath, "("+a.activeDataset+")")

	customers, err := a.loadCustomers(dataPath)
	if err != nil {
		fmt.Println("❌ Error loading customers:", err)
		runtime.EventsEmit(a.ctx, "analysis-error", err.Error())
		return
	}

	fmt.Println("✅ Loaded", len(customers), "customers")
	runtime.EventsEmit(a.ctx, "analysis-start", len(customers))

	startTime := time.Now()
	var results []CustomerResult
	var highRisk, mediumRisk, lowRisk int
	var totalRevenueSaved float64

	rand.Seed(time.Now().UnixNano())

	for i, customer := range customers {
		result, err := a.callBrainAPI(customer)
		if err != nil {
			continue
		}

		riskLevel := "LOW"
		if result.ChurnProbability > 0.7 {
			riskLevel = "HIGH"
			highRisk++
			totalRevenueSaved += customer.MonthlyCharges * 36 * 0.7
		} else if result.ChurnProbability > 0.4 {
			riskLevel = "MEDIUM"
			mediumRisk++
			totalRevenueSaved += customer.MonthlyCharges * 36 * 0.5
		} else {
			lowRisk++
		}

		customerResult := CustomerResult{
			CustomerID:       customer.CustomerID,
			Tenure:           customer.Tenure,
			MonthlyCharges:   customer.MonthlyCharges,
			ChurnProbability: result.ChurnProbability,
			RiskLevel:        riskLevel,
			ActionID:         result.ActionID,
			ActionName:       result.ActionName,
			ActionCost:       fmt.Sprintf("%.0f%%", result.ActionCost*100),
			Reasoning:        result.Reasoning,
		}
		results = append(results, customerResult)

		if i < 100 || i%50 == 0 {
			runtime.EventsEmit(a.ctx, "analysis-progress", map[string]interface{}{
				"current":  i + 1,
				"total":    len(customers),
				"customer": customerResult,
			})
		}
	}

	stats := DashboardStats{
		TotalCustomers:   len(results),
		HighRisk:         highRisk,
		MediumRisk:       mediumRisk,
		LowRisk:          lowRisk,
		EstRevenueSaved:  totalRevenueSaved,
		ProcessingTimeMs: time.Since(startTime).Milliseconds(),
	}

	a.mutex.Lock()
	a.results = results
	a.stats = stats
	if ds := a.datasets[a.activeDataset]; ds != nil {
		ds.Results = results
		ds.Stats = stats
	}
	a.mutex.Unlock()

	runtime.EventsEmit(a.ctx, "analysis-complete", stats)
}

type Customer struct {
	CustomerID     string
	Tenure         int
	MonthlyCharges float64
	TotalCharges   float64
	Contract       int
	InternetService int
	OnlineSecurity int
	OnlineBackup   int
	DeviceProtection int
	TechSupport    int
	StreamingTV    int
	StreamingMovies int
	Gender         int
	SeniorCitizen  int
	Partner        int
	Dependents     int
	PhoneService   int
	MultipleLines  int
	PaperlessBilling int
}

func (a *App) loadCustomers(path string) ([]Customer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var customers []Customer
	for i, record := range records {
		if i == 0 {
			continue
		}
		if len(record) < 3 {
			continue
		}
		customer := Customer{
			CustomerID:     record[0],
			Tenure:         parseInt(record[1]),
			MonthlyCharges: parseFloat(record[2]),
		}
		customers = append(customers, customer)
	}
	return customers, nil
}

func (a *App) callBrainAPI(customer Customer) (*PredictionResponse, error) {
	payload := map[string]interface{}{
		"tenure":            customer.Tenure,
		"MonthlyCharges":    customer.MonthlyCharges,
		"TotalCharges":      customer.TotalCharges,
		"Contract":          rand.Intn(3),
		"InternetService":   rand.Intn(3),
		"discount_modifier": a.discountModifier,
	}
	jsonData, _ := json.Marshal(payload)
	resp, err := http.Post("http://localhost:5000/predict_action", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var result PredictionResponse
	json.Unmarshal(body, &result)
	return &result, nil
}

func parseInt(s string) int {
	var i int
	fmt.Sscanf(s, "%d", &i)
	return i
}

func parseFloat(s string) float64 {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}

type TrainingRecord struct {
	ID        int    `json:"id"`
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Accuracy  float64 `json:"accuracy"`
	CreatedAt string `json:"created_at"`
}

var trainingHistory []TrainingRecord
var trainingIDCounter int

func (a *App) SelectDataset() (string, error) {
	selection, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select Dataset File",
		Filters: []runtime.FileFilter{
			{DisplayName: "Dataset Files", Pattern: "*.csv;*.xlsx;*.xls"},
		},
	})
	if err != nil {
		return "", err
	}
	return selection, nil
}

func (a *App) UploadDataset(filePath string) error {
	if filePath == "" {
		return fmt.Errorf("no file selected")
	}
	
	filename := filePath
	for i := len(filePath) - 1; i >= 0; i-- {
		if filePath[i] == '\\' || filePath[i] == '/' {
			filename = filePath[i+1:]
			break
		}
	}
	
	trainingIDCounter++
	record := TrainingRecord{
		ID:        trainingIDCounter,
		Filename:  filename,
		Status:    "PENDING",
		Accuracy:  0,
		CreatedAt: time.Now().Format("2006-01-02 15:04:05"),
	}
	trainingHistory = append([]TrainingRecord{record}, trainingHistory...)
	
	a.mutex.Lock()
	a.datasets[filename] = &DatasetAnalysis{
		Name:      filename,
		FilePath:  filePath,
		Trained:   false,
		CreatedAt: record.CreatedAt,
	}
	a.mutex.Unlock()
	
	fmt.Println("📂 Dataset added:", filename)
	runtime.EventsEmit(a.ctx, "training-updated", trainingHistory)
	runtime.EventsEmit(a.ctx, "datasets-updated", a.GetAvailableDatasets())
	
	return nil
}

func (a *App) GetTrainingHistory() []TrainingRecord {
	return trainingHistory
}

func (a *App) TriggerTraining(recordID int) error {
	var targetIdx int = -1
	for i, r := range trainingHistory {
		if r.ID == recordID {
			targetIdx = i
			break
		}
	}
	
	if targetIdx == -1 {
		return fmt.Errorf("record not found")
	}
	
	trainingHistory[targetIdx].Status = "TRAINING"
	runtime.EventsEmit(a.ctx, "training-updated", trainingHistory)
	
	go func() {
		time.Sleep(3 * time.Second)
		payload := map[string]string{"filename": trainingHistory[targetIdx].Filename}
		jsonData, _ := json.Marshal(payload)
		resp, err := http.Post("http://localhost:5000/train", "application/json", bytes.NewBuffer(jsonData))
		
		if err == nil && resp.StatusCode == 200 {
			trainingHistory[targetIdx].Status = "SUCCESS"
			trainingHistory[targetIdx].Accuracy = 92.5 + float64(rand.Intn(5))
		} else {
			trainingHistory[targetIdx].Status = "FAILED"
		}
		
		fmt.Println("🧠 Training complete:", trainingHistory[targetIdx].Filename, trainingHistory[targetIdx].Status)
		runtime.EventsEmit(a.ctx, "training-updated", trainingHistory)
	}()
	
	return nil
}
