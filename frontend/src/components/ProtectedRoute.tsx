import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useUserStore } from '../stores/userStore';

const ProtectedRoute: React.FC = () => {
    const { token } = useUserStore();

    if (!token) {
        return <Navigate to="/login" replace />;
    }

    return <Outlet />;
};

export default ProtectedRoute;
