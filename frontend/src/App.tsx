import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import MainLayout from './components/MainLayout';

import ProjectPage from './pages/ProjectPage';
import DatasetPage from './pages/DatasetPage';
import FeaturePage from './pages/FeaturePage';
import ModelingPage from './pages/ModelingPage';
import ResultPage from './pages/ResultPage';
import MonitorPage from './pages/MonitorPage';
import StrategyPage from './pages/StrategyPage';

import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import ProfilePage from './pages/ProfilePage';
import AdminUserPage from './pages/AdminUserPage';

const App: React.FC = () => {
    return (
        <ConfigProvider locale={zhCN}>
            <BrowserRouter>
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

                            <Route path="/profile" element={<ProfilePage />} />
                            <Route path="/admin/users" element={<AdminUserPage />} />
                        </Route>
                    </Route>
                </Routes>
            </BrowserRouter>
        </ConfigProvider>
    );
};

export default App;
