import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider, Spin } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import MainLayout from './components/MainLayout';

// 使用 lazy 动态导入页面组件
const ProjectPage = lazy(() => import('./pages/ProjectPage'));
const DatasetPage = lazy(() => import('./pages/DatasetPage'));
const FeaturePage = lazy(() => import('./pages/FeaturePage'));
const ModelingPage = lazy(() => import('./pages/ModelingPage'));
const ResultPage = lazy(() => import('./pages/ResultPage'));
const MonitorPage = lazy(() => import('./pages/MonitorPage'));
const StrategyPage = lazy(() => import('./pages/StrategyPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const AdminUserPage = lazy(() => import('./pages/AdminUserPage'));
const AgentPage = lazy(() => import('./pages/AgentPage'));

import ProtectedRoute from './components/ProtectedRoute';

const LoadingFallback = () => (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" tip="加载中..." />
    </div>
);

const App: React.FC = () => {
    return (
        <ConfigProvider locale={zhCN}>
            <BrowserRouter>
                <Suspense fallback={<LoadingFallback />}>
                    <Routes>
                        <Route path="/login" element={<LoginPage />} />

                        <Route element={<ProtectedRoute />}>
                            <Route path="/" element={<MainLayout />}>
                                <Route index element={<ProjectPage />} />
                                <Route path="dataset" element={<DatasetPage />} />
                                <Route path="feature" element={<FeaturePage />} />
                                <Route path="/modeling" element={<ModelingPage />} />
                                <Route path="/results" element={<ResultPage />} />
                                <Route path="/strategy" element={<StrategyPage />} />
                                <Route path="/monitor" element={<MonitorPage />} />
                                <Route path="/agent" element={<AgentPage />} />

                                <Route path="/profile" element={<ProfilePage />} />
                                <Route path="/admin/users" element={<AdminUserPage />} />
                            </Route>
                        </Route>
                    </Routes>
                </Suspense>
            </BrowserRouter>
        </ConfigProvider>
    );
};

export default App;

