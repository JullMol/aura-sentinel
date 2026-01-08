package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type PredictionRequest struct {
	Tenure            int     `json:"tenure"`
	MonthlyCharges    float64 `json:"MonthlyCharges"`
	TotalCharges      float64 `json:"TotalCharges"`
	Contract          int     `json:"Contract"`
	InternetService   int     `json:"InternetService"`
	OnlineSecurity    int     `json:"OnlineSecurity"`
	OnlineBackup      int     `json:"OnlineBackup"`
	DeviceProtection  int     `json:"DeviceProtection"`
	TechSupport       int     `json:"TechSupport"`
	StreamingTV       int     `json:"StreamingTV"`
	StreamingMovies   int     `json:"StreamingMovies"`
	Gender            int     `json:"gender"`
	SeniorCitizen     int     `json:"SeniorCitizen"`
	Partner           int     `json:"Partner"`
	Dependents        int     `json:"Dependents"`
	PhoneService      int     `json:"PhoneService"`
	MultipleLines     int     `json:"MultipleLines"`
	PaperlessBilling  int     `json:"PaperlessBilling"`
	PrevInterventions int     `json:"prev_interventions"`
	DaysSinceContact  int     `json:"days_since_contact"`
	ResponseRate      float64 `json:"response_rate"`
	EngagementScore   float64 `json:"engagement_score"`
}

type PredictionResponse struct {
	ChurnProbability float64 `json:"churn_probability"`
	ActionID         int     `json:"action_id"`
	ActionName       string  `json:"action_name"`
	ActionCost       float64 `json:"action_cost"`
	Reasoning        string  `json:"reasoning"`
	Status           string  `json:"status"`
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
}

type StreamMessage struct {
	Type             string  `json:"type"`
	CustomerID       string  `json:"customer_id"`
	Tenure           int     `json:"tenure"`
	MonthlyCharges   float64 `json:"monthly_charges"`
	ChurnProbability float64 `json:"churn_probability"`
	RiskLevel        string  `json:"risk_level"`
	Action           string  `json:"action"`
	ActionCost       string  `json:"action_cost"`
	Reasoning        string  `json:"reasoning"`
	Timestamp        string  `json:"timestamp"`
	Progress         int     `json:"progress"`
	Total            int     `json:"total"`
}

type DashboardStats struct {
	TotalCustomers   int     `json:"total_customers"`
	HighRisk         int     `json:"high_risk"`
	MediumRisk       int     `json:"medium_risk"`
	LowRisk          int     `json:"low_risk"`
	AvgChurnProb     float64 `json:"avg_churn_prob"`
	EstRevenueSaved  float64 `json:"est_revenue_saved"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

type APIResponse struct {
	Stats     DashboardStats   `json:"stats"`
	Customers []CustomerResult `json:"customers"`
}

type Hub struct {
	clients   map[*websocket.Conn]bool
	broadcast chan interface{}
	mutex     sync.RWMutex
}

var hub = Hub{
	clients:   make(map[*websocket.Conn]bool),
	broadcast: make(chan interface{}, 100),
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

const (
	BrainAPIURL = "http://localhost:5000/predict_action"
	DataPath    = "../../data/dataset.xls"
	OutputPath  = "retention_strategy_results.csv"
)

var (
	cachedResults    []CustomerResult
	cachedStats      DashboardStats
	cacheMutex       sync.RWMutex
	cacheReady       bool
	streamActive     bool
	discountModifier float64 = 1.0
	neighborNode     string  = "http://localhost:8081"
)

func (h *Hub) Run() {
	for msg := range h.broadcast {
		h.mutex.RLock()
		for client := range h.clients {
			err := client.WriteJSON(msg)
			if err != nil {
				client.Close()
				h.mutex.RUnlock()
				h.mutex.Lock()
				delete(h.clients, client)
				h.mutex.Unlock()
				h.mutex.RLock()
			}
		}
		h.mutex.RUnlock()
	}
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}

	hub.mutex.Lock()
	hub.clients[conn] = true
	hub.mutex.Unlock()

	fmt.Println("🖥️  Dashboard connected to Matrix Stream")

	hub.broadcast <- map[string]interface{}{
		"type":    "CONNECTED",
		"message": "Matrix Stream Active",
	}

	go func() {
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				break
			}
			var cmd map[string]interface{}
			json.Unmarshal(msg, &cmd)

			if cmd["type"] == "ORACLE_UPDATE" {
				if val, ok := cmd["value"].(float64); ok {
					discountModifier = val
					fmt.Printf("🔮 Oracle Mode: Discount Modifier updated to %.1fx\n", discountModifier)
					hub.broadcast <- map[string]interface{}{
						"type":    "ORACLE_ACK",
						"value":   discountModifier,
						"message": fmt.Sprintf("Oracle updated to %.1fx", discountModifier),
					}
				}
			}
		}
	}()
}

func handleStartStream(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if streamActive {
		json.NewEncoder(w).Encode(map[string]string{"status": "already_running"})
		return
	}
	go startMatrixStream()
	json.NewEncoder(w).Encode(map[string]string{"status": "started"})
}

func startMatrixStream() {
	if streamActive {
		return
	}
	streamActive = true

	customers, err := ProcessExcel(DataPath)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		streamActive = false
		return
	}

	hub.broadcast <- map[string]interface{}{
		"type":    "STREAM_START",
		"total":   len(customers),
		"message": "Matrix Stream Initialized",
	}

	rand.Seed(time.Now().UnixNano())
	startTime := time.Now()

	var results []CustomerResult
	var highRisk, mediumRisk, lowRisk int
	var totalChurnProb, totalRevenueSaved float64

	for i, customer := range customers {
		req := PredictionRequest{
			Tenure:            customer.Tenure,
			MonthlyCharges:    customer.MonthlyCharges,
			TotalCharges:      customer.TotalCharges,
			Contract:          customer.Contract,
			InternetService:   customer.InternetService,
			OnlineSecurity:    customer.OnlineSecurity,
			OnlineBackup:      customer.OnlineBackup,
			DeviceProtection:  customer.DeviceProtection,
			TechSupport:       customer.TechSupport,
			StreamingTV:       customer.StreamingTV,
			StreamingMovies:   customer.StreamingMovies,
			Gender:            customer.Gender,
			SeniorCitizen:     customer.SeniorCitizen,
			Partner:           customer.Partner,
			Dependents:        customer.Dependents,
			PhoneService:      customer.PhoneService,
			MultipleLines:     customer.MultipleLines,
			PaperlessBilling:  customer.PaperlessBilling,
			PrevInterventions: rand.Intn(4),
			DaysSinceContact:  rand.Intn(60),
			ResponseRate:      rand.Float64() * 0.5,
			EngagementScore:   30 + rand.Float64()*50,
		}

		result, err := callBrainAPI(req)
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

		totalChurnProb += result.ChurnProbability

		customerResult := CustomerResult{
			CustomerID:       customer.CustomerID,
			Tenure:           customer.Tenure,
			MonthlyCharges:   customer.MonthlyCharges,
			ChurnProbability: result.ChurnProbability,
			RiskLevel:        riskLevel,
			ActionID:         result.ActionID,
			ActionName:       result.ActionName,
			ActionCost:       fmt.Sprintf("%.0f%%", result.ActionCost*100),
		}
		results = append(results, customerResult)

		if i < 100 || i%50 == 0 {
			hub.broadcast <- StreamMessage{
				Type:             "STREAM_DATA",
				CustomerID:       customer.CustomerID,
				Tenure:           customer.Tenure,
				MonthlyCharges:   customer.MonthlyCharges,
				ChurnProbability: result.ChurnProbability,
				RiskLevel:        riskLevel,
				Action:           result.ActionName,
				ActionCost:       fmt.Sprintf("%.0f%%", result.ActionCost*100),
				Reasoning:        result.Reasoning,
				Timestamp:        time.Now().Format("15:04:05"),
				Progress:         i + 1,
				Total:            len(customers),
			}

			if i < 100 {
				time.Sleep(50 * time.Millisecond)
			}
		}
	}

	elapsed := time.Since(startTime)

	saveResultsToCSV(results)

	cacheMutex.Lock()
	cachedResults = results
	cachedStats = DashboardStats{
		TotalCustomers:   len(results),
		HighRisk:         highRisk,
		MediumRisk:       mediumRisk,
		LowRisk:          lowRisk,
		AvgChurnProb:     totalChurnProb / float64(len(results)) * 100,
		EstRevenueSaved:  totalRevenueSaved,
		ProcessingTimeMs: elapsed.Milliseconds(),
	}
	cacheReady = true
	cacheMutex.Unlock()

	hub.broadcast <- map[string]interface{}{
		"type":            "STREAM_COMPLETE",
		"total_processed": len(results),
		"high_risk":       highRisk,
		"medium_risk":     mediumRisk,
		"low_risk":        lowRisk,
		"revenue_saved":   totalRevenueSaved,
		"time_ms":         elapsed.Milliseconds(),
	}

	fmt.Printf("\n✅ Stream complete! %d customers processed\n", len(results))
	streamActive = false
}

func main() {
	fmt.Println("🚀 Aura-Sentinel Engine v2.0: Enterprise Mode")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	if err := InitDatabase(); err != nil {
		fmt.Printf("⚠️ Database error: %v\n", err)
	}

	go hub.Run()

	http.HandleFunc("/ws", handleWS)
	http.HandleFunc("/api/start-stream", handleStartStream)
	http.HandleFunc("/api/results", handleGetResults)
	http.HandleFunc("/api/stats", handleGetStats)
	http.HandleFunc("/api/sync", handleSync)
	http.HandleFunc("/api/upload", handleUpload)
	http.HandleFunc("/api/train", handleTrain)
	http.HandleFunc("/api/history", handleHistory)
	http.HandleFunc("/health", handleHealth)

	fmt.Println("\n📡 Endpoints:")
	fmt.Println("   WebSocket:  ws://localhost:8080/ws")
	fmt.Println("   Stream:     GET  /api/start-stream")
	fmt.Println("   Results:    GET  /api/results")
	fmt.Println("   Upload:     POST /api/upload")
	fmt.Println("   Train:      POST /api/train")
	fmt.Println("   History:    GET  /api/history")
	fmt.Println("\n⏳ Waiting for dashboard connection...")

	log.Fatal(http.ListenAndServe(":8080", nil))
}

func enableCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "healthy",
		"ready":         cacheReady,
		"stream_active": streamActive,
	})
}

func handleGetStats(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	w.Header().Set("Content-Type", "application/json")
	cacheMutex.RLock()
	defer cacheMutex.RUnlock()
	if !cacheReady {
		json.NewEncoder(w).Encode(map[string]string{"status": "processing"})
		return
	}
	json.NewEncoder(w).Encode(cachedStats)
}

func handleGetResults(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	w.Header().Set("Content-Type", "application/json")
	cacheMutex.RLock()
	defer cacheMutex.RUnlock()
	if !cacheReady {
		json.NewEncoder(w).Encode(map[string]string{"status": "processing"})
		return
	}
	limit := 100
	results := cachedResults
	if len(results) > limit {
		results = results[:limit]
	}
	json.NewEncoder(w).Encode(APIResponse{Stats: cachedStats, Customers: results})
}

func saveResultsToCSV(results []CustomerResult) {
	file, err := os.Create(OutputPath)
	if err != nil {
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	writer.Write([]string{"CustomerID", "Tenure", "MonthlyCharges", "ChurnProbability", "RiskLevel", "ActionID", "RecommendedAction", "ActionCost"})
	for _, r := range results {
		writer.Write([]string{r.CustomerID, fmt.Sprintf("%d", r.Tenure), fmt.Sprintf("%.2f", r.MonthlyCharges), fmt.Sprintf("%.4f", r.ChurnProbability), r.RiskLevel, fmt.Sprintf("%d", r.ActionID), r.ActionName, r.ActionCost})
	}
}

func callBrainAPI(req PredictionRequest) (*PredictionResponse, error) {
	payload := map[string]interface{}{
		"tenure":             req.Tenure,
		"MonthlyCharges":     req.MonthlyCharges,
		"TotalCharges":       req.TotalCharges,
		"Contract":           req.Contract,
		"InternetService":    req.InternetService,
		"OnlineSecurity":     req.OnlineSecurity,
		"OnlineBackup":       req.OnlineBackup,
		"DeviceProtection":   req.DeviceProtection,
		"TechSupport":        req.TechSupport,
		"StreamingTV":        req.StreamingTV,
		"StreamingMovies":    req.StreamingMovies,
		"gender":             req.Gender,
		"SeniorCitizen":      req.SeniorCitizen,
		"Partner":            req.Partner,
		"Dependents":         req.Dependents,
		"PhoneService":       req.PhoneService,
		"MultipleLines":      req.MultipleLines,
		"PaperlessBilling":   req.PaperlessBilling,
		"prev_interventions": req.PrevInterventions,
		"days_since_contact": req.DaysSinceContact,
		"response_rate":      req.ResponseRate,
		"engagement_score":   req.EngagementScore,
		"discount_modifier":  discountModifier,
	}
	jsonData, _ := json.Marshal(payload)
	resp, err := http.Post(BrainAPIURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var result PredictionResponse
	json.Unmarshal(body, &result)
	return &result, nil
}

func broadcastKnowledge(customerID string, action string, profit float64) {
	payload := map[string]interface{}{
		"type":        "KNOWLEDGE_SHARE",
		"customer_id": customerID,
		"best_action": action,
		"profit_gain": profit,
	}
	jsonData, _ := json.Marshal(payload)
	go func() {
		http.Post(neighborNode+"/api/sync", "application/json", bytes.NewBuffer(jsonData))
	}()
}

func handleSync(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	var knowledge map[string]interface{}
	json.NewDecoder(r.Body).Decode(&knowledge)

	bestAction := ""
	if action, ok := knowledge["best_action"].(string); ok {
		bestAction = action
	}

	fmt.Printf("🤝 RECEIVED KNOWLEDGE from neighbor: Action '%s' is profitable!\n", bestAction)

	hub.broadcast <- map[string]interface{}{
		"type":    "NODE_SYNC",
		"message": fmt.Sprintf("📡 Syncing: %s recommended from neighbor node", bestAction),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "synced"})
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == "OPTIONS" {
		return
	}

	r.ParseMultipartForm(10 << 20)
	file, handler, err := r.FormFile("dataset")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "File upload failed"})
		return
	}
	defer file.Close()

	os.MkdirAll("../../data/uploads", os.ModePerm)
	dst, err := os.Create("../../data/uploads/" + handler.Filename)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to save file"})
		return
	}
	defer dst.Close()
	io.Copy(dst, file)

	AddTrainingRecord(handler.Filename)

	fmt.Printf("📤 File uploaded: %s\n", handler.Filename)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": fmt.Sprintf("File %s uploaded successfully", handler.Filename),
	})
}

func handleTrain(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == "OPTIONS" {
		return
	}

	var req struct {
		Filename string `json:"filename"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	fmt.Printf("🧪 Training requested for: %s\n", req.Filename)

	go func() {
		resp, err := http.Post("http://localhost:5000/train", "application/json",
			bytes.NewBuffer([]byte(fmt.Sprintf(`{"filename": "%s"}`, req.Filename))))
		if err != nil {
			UpdateTrainingStatus(req.Filename, "FAILED", 0)
			return
		}
		defer resp.Body.Close()

		var result map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&result)

		accuracy := 0
		if acc, ok := result["accuracy"].(float64); ok {
			accuracy = int(acc)
		}
		UpdateTrainingStatus(req.Filename, "SUCCESS", accuracy)

		hub.broadcast <- map[string]interface{}{
			"type":     "TRAINING_COMPLETE",
			"filename": req.Filename,
			"accuracy": accuracy,
		}
	}()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "started",
		"message": "Training started in background",
	})
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	w.Header().Set("Content-Type", "application/json")

	records, err := GetTrainingHistory()
	if err != nil {
		json.NewEncoder(w).Encode([]TrainingRecord{})
		return
	}

	if records == nil {
		records = []TrainingRecord{}
	}
	json.NewEncoder(w).Encode(records)
}

