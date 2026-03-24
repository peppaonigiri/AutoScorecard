import React from 'react';
import { Layout, Menu, Typography } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
    ProjectOutlined,
    DatabaseOutlined,
    BarChartOutlined,
    GatewayOutlined,
    RocketOutlined,
    LineChartOutlined,
    AreaChartOutlined
} from '@ant-design/icons';
import { useAppStore } from '../stores';
import { useUserStore } from '../stores/userStore';
import { Dropdown } from 'antd';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

const MainLayout: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const currentProjectId = useAppStore((state) => state.currentProjectId);
    const { user, logout } = useUserStore();

    const menuItems = [
        {
            key: '/',
            icon: <ProjectOutlined />,
            label: '项目管理',
        },
        {
            key: '/dataset',
            icon: <DatabaseOutlined />,
            label: '数据管理',
            disabled: !currentProjectId,
        },
        {
            key: '/feature',
            icon: <BarChartOutlined />,
            label: '变量分析',
            disabled: !currentProjectId,
        },
        {
            key: '/modeling',
            icon: <RocketOutlined />,
            label: '模型训练',
            disabled: !currentProjectId,
        },
        {
            key: '/results',
            icon: <GatewayOutlined />,
            label: '模型结果',
            disabled: !currentProjectId,
        },
        {
            key: '/strategy',
            icon: <AreaChartOutlined />,
            label: '策略制定',
            disabled: !currentProjectId,
        },
        {
            key: '/monitor',
            icon: <LineChartOutlined />,
            label: '上线监控',
            disabled: !currentProjectId,
        },
    ];

    if (user?.is_admin) {
        menuItems.push({
            key: '/admin/users',
            icon: <GatewayOutlined />, // Using a generic icon
            label: '用户管理',
            disabled: false,
        });
    }

    const userMenu = {
        items: [
            {
                key: 'profile',
                label: '个人中心',
                onClick: () => navigate('/profile')
            },
            {
                key: 'logout',
                label: '退出登录',
                onClick: logout
            }
        ]
    };

    return (
        <Layout style={{ minHeight: '100vh' }}>
            <Header style={{ display: 'flex', alignItems: 'center', background: '#001529', padding: '0 24px' }}>
                <Title level={3} style={{ color: '#fff', margin: 0 }}>AutoModeling Platform</Title>
                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
                    {currentProjectId && (
                        <div style={{ color: '#fff' }}>
                            当前项目 ID: {currentProjectId}
                        </div>
                    )}
                    {user && (
                        <Dropdown menu={userMenu} placement="bottomRight">
                            <div style={{ color: '#fff', cursor: 'pointer', padding: '0 12px' }}>
                                Hi, {user.username} {user.is_admin ? '(Admin)' : ''}
                            </div>
                        </Dropdown>
                    )}
                </div>
            </Header>
            <Layout>
                <Sider width={200} theme="light">
                    <Menu
                        mode="inline"
                        selectedKeys={[location.pathname]}
                        style={{ height: '100%', borderRight: 0 }}
                        items={menuItems}
                        onClick={({ key }) => navigate(key)}
                    />
                </Sider>
                <Layout style={{ padding: '24px' }}>
                    <Content
                        style={{
                            padding: 24,
                            margin: 0,
                            minHeight: 280,
                            background: '#fff',
                            borderRadius: 8,
                        }}
                    >
                        <Outlet />
                    </Content>
                </Layout>
            </Layout>
        </Layout>
    );
};

export default MainLayout;
