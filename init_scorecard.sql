-- AutoModeling PostgreSQL Database Initialization Script

-- 1. Create Base Tables
DROP TABLE IF EXISTS strategy_monitoring_logs;
DROP TABLE IF EXISTS strategies;
DROP TABLE IF EXISTS monitoring_logs;
DROP TABLE IF EXISTS deployments;
DROP TABLE IF EXISTS model_results;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS datasets;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS users;

-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(200) NOT NULL,
    is_admin INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Projects Table
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT DEFAULT '',
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feature_list JSONB DEFAULT '[]',
    split_config JSONB DEFAULT '{}'
);

-- Datasets Table
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT DEFAULT 0,
    n_rows INTEGER DEFAULT 0,
    n_cols INTEGER DEFAULT 0,
    columns_info JSONB DEFAULT '{}',
    stats_cache JSONB DEFAULT '{}',
    l1_results JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks Table
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    params JSONB DEFAULT '{}',
    result JSONB DEFAULT '{}',
    error_msg TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model Results Table
CREATE TABLE model_results (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    model_type VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) DEFAULT '',
    params JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    feature_list JSONB DEFAULT '[]',
    optuna_strategy INTEGER,
    n_trials INTEGER DEFAULT 0,
    score_config JSONB DEFAULT '{}',
    score_distribution JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Deployment Table
CREATE TABLE deployments (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    model_result_id INTEGER REFERENCES model_results(id),
    status VARCHAR(50) DEFAULT 'active',
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Monitoring Logs Table
CREATE TABLE monitoring_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    deployment_id INTEGER REFERENCES deployments(id),
    batch_name VARCHAR(100),
    sample_size INTEGER DEFAULT 0,
    psi FLOAT DEFAULT 0.0,
    avg_score FLOAT DEFAULT 0.0,
    score_dist JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Strategies Table
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',
    status VARCHAR(50) DEFAULT 'draft',
    priority INTEGER DEFAULT 0,
    combine_logic VARCHAR(20) DEFAULT 'and',
    rule_type VARCHAR(20) DEFAULT 'reject',
    rules JSONB DEFAULT '[]',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Strategy Monitoring Logs Table
CREATE TABLE strategy_monitoring_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    batch_name VARCHAR(100),
    total_count INTEGER DEFAULT 0,
    pass_count INTEGER DEFAULT 0,
    hit_count INTEGER DEFAULT 0,
    approval_rate FLOAT DEFAULT 0.0,
    rule_stats JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Seed Admin Data (Password: root)
INSERT INTO users (username, hashed_password, is_admin) 
VALUES ('root', '$2b$12$N9qo8uLOickgx2ZMRZoMyeIjZAgNo3gKqX.Jc8p30Z2YwTzjx/3Y6', 1);

-- 3. Create Basic Info
COMMENT ON DATABASE scorecard IS 'AutoModeling Persistence Store';
