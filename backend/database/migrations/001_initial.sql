-- 001_initial.sql
-- Initial database schema for City Guardian Environmental Intelligence System

-- Enable PostGIS (if available)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- Reports table
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(20) UNIQUE NOT NULL,
    domain VARCHAR(50) NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    severity_score FLOAT NOT NULL,
    status VARCHAR(20) DEFAULT 'OPEN',
    timestamp TIMESTAMP DEFAULT NOW(),
    image_path TEXT,
    annotated_path TEXT,
    ai_context TEXT,
    responding_department VARCHAR(200),
    recommended_action TEXT,
    total_detections INTEGER DEFAULT 0
);

-- Incidents table (detected objects per report)
CREATE TABLE IF NOT EXISTS incidents (
    id SERIAL PRIMARY KEY,
    report_db_id INTEGER REFERENCES reports(id) ON DELETE CASCADE,
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT,
    bbox_y1 FLOAT,
    bbox_x2 FLOAT,
    bbox_y2 FLOAT,
    area_fraction FLOAT
);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    report_db_id INTEGER REFERENCES reports(id) ON DELETE CASCADE,
    latitude FLOAT,
    longitude FLOAT,
    display_address TEXT,
    ward VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(100),
    postcode VARCHAR(20),
    country VARCHAR(100)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_reports_domain ON reports(domain);
CREATE INDEX IF NOT EXISTS idx_reports_severity ON reports(severity_level);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON reports(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_locations_latlon ON locations(latitude, longitude);
