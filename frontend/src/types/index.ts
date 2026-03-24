export interface Project {
    id: number;
    name: string;
    description?: string;
    status: string;
    created_at: string;
    updated_at: string;
}

export interface Dataset {
    id: number;
    project_id: number;
    name: string;
    file_path: string;
    rows?: number;
    cols?: number;
    created_at: string;
}

export interface Task {
    id: number;
    project_id: number;
    task_type: string;
    status: string;
    progress: number;
    params: any;
    result: any;
    error_msg?: string;
    created_at: string;
    updated_at: string;
}
