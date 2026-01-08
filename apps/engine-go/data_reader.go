package main

import (
	"encoding/csv"
	"os"
	"strconv"
)

type Customer struct {
	CustomerID       string  `json:"customerID"`
	Gender           int     `json:"gender"`
	SeniorCitizen    int     `json:"SeniorCitizen"`
	Partner          int     `json:"Partner"`
	Dependents       int     `json:"Dependents"`
	Tenure           int     `json:"tenure"`
	PhoneService     int     `json:"PhoneService"`
	MultipleLines    int     `json:"MultipleLines"`
	InternetService  int     `json:"InternetService"`
	OnlineSecurity   int     `json:"OnlineSecurity"`
	OnlineBackup     int     `json:"OnlineBackup"`
	DeviceProtection int     `json:"DeviceProtection"`
	TechSupport      int     `json:"TechSupport"`
	StreamingTV      int     `json:"StreamingTV"`
	StreamingMovies  int     `json:"StreamingMovies"`
	Contract         int     `json:"Contract"`
	PaperlessBilling int     `json:"PaperlessBilling"`
	MonthlyCharges   float64 `json:"MonthlyCharges"`
	TotalCharges     float64 `json:"TotalCharges"`
}

func stringToInt(s string) int {
	val, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return val
}

func stringToFloat(s string) float64 {
	val, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0.0
	}
	return val
}

func mapBinary(val string) int {
	switch val {
	case "Yes", "Male":
		return 1
	case "No", "Female":
		return 0
	default:
		return 0
	}
}

func mapMultipleLines(val string) int {
	switch val {
	case "Yes":
		return 2
	case "No":
		return 1
	case "No phone service":
		return 0
	default:
		return 0
	}
}

func mapInternetService(val string) int {
	switch val {
	case "Fiber optic":
		return 2
	case "DSL":
		return 1
	case "No":
		return 0
	default:
		return 0
	}
}

func mapServiceFeature(val string) int {
	switch val {
	case "Yes":
		return 2
	case "No":
		return 1
	case "No internet service":
		return 0
	default:
		return 0
	}
}

func mapContract(val string) int {
	switch val {
	case "Month-to-month":
		return 0
	case "One year":
		return 1
	case "Two year":
		return 2
	default:
		return 0
	}
}

func ProcessExcel(filePath string) ([]Customer, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var customers []Customer
	for i, row := range rows {
		if i == 0 || len(row) < 20 {
			continue
		}

		cust := Customer{
			CustomerID:       row[0],
			Gender:           mapBinary(row[1]),
			SeniorCitizen:    stringToInt(row[2]),
			Partner:          mapBinary(row[3]),
			Dependents:       mapBinary(row[4]),
			Tenure:           stringToInt(row[5]),
			PhoneService:     mapBinary(row[6]),
			MultipleLines:    mapMultipleLines(row[7]),
			InternetService:  mapInternetService(row[8]),
			OnlineSecurity:   mapServiceFeature(row[9]),
			OnlineBackup:     mapServiceFeature(row[10]),
			DeviceProtection: mapServiceFeature(row[11]),
			TechSupport:      mapServiceFeature(row[12]),
			StreamingTV:      mapServiceFeature(row[13]),
			StreamingMovies:  mapServiceFeature(row[14]),
			Contract:         mapContract(row[15]),
			PaperlessBilling: mapBinary(row[16]),
			MonthlyCharges:   stringToFloat(row[18]),
			TotalCharges:     stringToFloat(row[19]),
		}
		customers = append(customers, cust)
	}

	return customers, nil
}