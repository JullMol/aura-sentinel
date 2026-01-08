package main

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type TrainingRecord struct {
	ID        int    `json:"id"`
	Filename  string `json:"filename"`
	Status    string `json:"status"`
	Accuracy  int    `json:"accuracy"`
	CreatedAt string `json:"created_at"`
}

var db *sql.DB

func InitDatabase() error {
	var err error
	db, err = sql.Open("sqlite3", "./training_history.db")
	if err != nil {
		return err
	}

	createTable := `
	CREATE TABLE IF NOT EXISTS training_history (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		filename TEXT NOT NULL,
		status TEXT DEFAULT 'PENDING',
		accuracy INTEGER DEFAULT 0,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = db.Exec(createTable)
	if err != nil {
		return err
	}

	fmt.Println("✅ SQLite database initialized")
	return nil
}

func AddTrainingRecord(filename string) (int64, error) {
	result, err := db.Exec(
		"INSERT INTO training_history (filename, status, created_at) VALUES (?, 'PENDING', ?)",
		filename, time.Now().Format("2006-01-02 15:04:05"),
	)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func GetTrainingHistory() ([]TrainingRecord, error) {
	rows, err := db.Query("SELECT id, filename, status, accuracy, created_at FROM training_history ORDER BY id DESC LIMIT 20")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []TrainingRecord
	for rows.Next() {
		var r TrainingRecord
		if err := rows.Scan(&r.ID, &r.Filename, &r.Status, &r.Accuracy, &r.CreatedAt); err != nil {
			continue
		}
		records = append(records, r)
	}
	return records, nil
}

func UpdateTrainingStatus(filename string, status string, accuracy int) error {
	_, err := db.Exec(
		"UPDATE training_history SET status = ?, accuracy = ? WHERE filename = ?",
		status, accuracy, filename,
	)
	return err
}
