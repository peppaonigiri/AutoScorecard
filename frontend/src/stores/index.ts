import { create } from 'zustand';

interface AppState {
    currentProjectId: number | null;
    setCurrentProjectId: (id: number | null) => void;

    // 数据集状态 (全局持久化，切换页面不丢失)
    currentDatasetId: number | null;
    setCurrentDatasetId: (id: number | null) => void;
    datasetInfo: any;
    setDatasetInfo: (info: any) => void;
    datasetStats: any;
    setDatasetStats: (stats: any) => void;
    previewData: any;
    setPreviewData: (data: any) => void;

    // Feature Page 缓存
    ivReport: any[];
    setIvReport: (report: any[]) => void;
    filterResult: any;
    setFilterResult: (result: any) => void;
    selectedFeatures: string[];
    setSelectedFeatures: (features: string[]) => void;

    // 全局排除特征
    excludeCols: string[];
    setExcludeCols: (cols: string[]) => void;
}

export const useAppStore = create<AppState>((set) => ({
    currentProjectId: null,
    setCurrentProjectId: (id) => set({ currentProjectId: id }),

    currentDatasetId: null,
    setCurrentDatasetId: (id) => set({ currentDatasetId: id }),
    datasetInfo: null,
    setDatasetInfo: (info) => set({ datasetInfo: info }),
    datasetStats: null,
    setDatasetStats: (stats) => set({ datasetStats: stats }),
    previewData: null,
    setPreviewData: (data) => set({ previewData: data }),

    ivReport: [],
    setIvReport: (report) => set({ ivReport: report }),
    filterResult: null,
    setFilterResult: (result) => set({ filterResult: result }),
    selectedFeatures: [],
    setSelectedFeatures: (features) => set({ selectedFeatures: features }),

    excludeCols: [],
    setExcludeCols: (cols) => set({ excludeCols: cols }),
}));
