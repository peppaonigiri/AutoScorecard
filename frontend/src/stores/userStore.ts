import { create } from 'zustand';

interface User {
    id: number;
    username: string;
    is_admin: number;
    is_active: number;
}

interface UserState {
    token: string | null;
    user: User | null;
    setToken: (token: string | null) => void;
    setUser: (user: User | null) => void;
    logout: () => void;
}

const getInitialToken = () => localStorage.getItem('access_token') || null;
const getInitialUser = () => {
    const userStr = localStorage.getItem('user_info');
    if (userStr) {
        try {
            return JSON.parse(userStr);
        } catch (e) {
            return null;
        }
    }
    return null;
};

export const useUserStore = create<UserState>((set) => ({
    token: getInitialToken(),
    user: getInitialUser(),
    setToken: (token) => {
        if (token) {
            localStorage.setItem('access_token', token);
        } else {
            localStorage.removeItem('access_token');
        }
        set({ token });
    },
    setUser: (user) => {
        if (user) {
            localStorage.setItem('user_info', JSON.stringify(user));
        } else {
            localStorage.removeItem('user_info');
        }
        set({ user });
    },
    logout: () => {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_info');
        set({ token: null, user: null });
        window.location.href = '/login';
    }
}));
