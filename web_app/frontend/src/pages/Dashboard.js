import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Baby,
  Mic,
  Users,
  Activity,
  TrendingUp,
  Clock,
  Brain,
  Heart,
  Zap,
  ArrowRight,
  RefreshCw
} from 'lucide-react';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [babies, setBabies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Load system status
      const statusResponse = await fetch('/api/system-status');
      const statusData = await statusResponse.json();
      setSystemStatus(statusData);

      // Load babies
      const babiesResponse = await fetch('/api/babies');
      const babiesData = await babiesResponse.json();
      setBabies(babiesData);

    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    {
      title: 'Analyze Cry',
      description: 'Upload or record a baby cry for instant AI analysis',
      icon: Mic,
      path: '/classify',
      color: 'bg-blue-500 hover:bg-blue-600',
      textColor: 'text-blue-700'
    },
    {
      title: 'Manage Babies',
      description: 'Add or update baby profiles and care history',
      icon: Users,
      path: '/babies',
      color: 'bg-green-500 hover:bg-green-600',
      textColor: 'text-green-700'
    },
    {
      title: 'View Analytics',
      description: 'Explore patterns and insights from cry analysis',
      icon: Activity,
      path: '/analytics',
      color: 'bg-purple-500 hover:bg-purple-600',
      textColor: 'text-purple-700'
    }
  ];

  const stats = [
    {
      label: 'Active Baby Profiles',
      value: babies.length,
      icon: Baby,
      color: 'text-baby-600',
      bgColor: 'bg-baby-50'
    },
    {
      label: 'Model Accuracy',
      value: '83.7%',
      icon: Brain,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      label: 'AI Features',
      value: '323',
      icon: Zap,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      label: 'System Status',
      value: systemStatus?.model_loaded ? 'Online' : 'Offline',
      icon: Heart,
      color: systemStatus?.model_loaded ? 'text-green-600' : 'text-red-600',
      bgColor: systemStatus?.model_loaded ? 'bg-green-50' : 'bg-red-50'
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex items-center space-x-3">
          <RefreshCw className="w-5 h-5 animate-spin text-baby-500" />
          <span className="text-gray-600">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Welcome to BabyWhisper
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          AI-powered baby cry analysis with context-aware intelligence. 
          Understanding your baby's needs through advanced audio processing and personalized insights.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const IconComponent = stat.icon;
          return (
            <div
              key={index}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">
                    {stat.label}
                  </p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stat.value}
                  </p>
                </div>
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <IconComponent className={`w-6 h-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action, index) => {
            const IconComponent = action.icon;
            return (
              <Link
                key={index}
                to={action.path}
                className="group bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-all hover:-translate-y-1"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-lg ${action.color} transition-colors`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600 transition-colors" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {action.title}
                </h3>
                <p className="text-gray-600 text-sm">
                  {action.description}
                </p>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent Activity / Baby Profiles Preview */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Baby Profiles</h2>
          <Link
            to="/babies"
            className="text-baby-600 hover:text-baby-700 text-sm font-medium flex items-center"
          >
            View All <ArrowRight className="w-4 h-4 ml-1" />
          </Link>
        </div>

        {babies.length > 0 ? (
          <div className="space-y-3">
            {babies.slice(0, 3).map((baby) => (
              <div
                key={baby.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-baby-100 rounded-full">
                    <Baby className="w-4 h-4 text-baby-600" />
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900">{baby.name}</h3>
                    <p className="text-sm text-gray-600">
                      {baby.age_months} months old
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-600 flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    {baby.last_feeding 
                      ? `Fed ${new Date(baby.last_feeding).toLocaleTimeString()}`
                      : 'No recent feeding'
                    }
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Baby className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Baby Profiles</h3>
            <p className="text-gray-600 mb-4">
              Create your first baby profile to start using context-aware cry analysis
            </p>
            <Link
              to="/babies"
              className="inline-flex items-center px-4 py-2 bg-baby-500 text-white rounded-lg hover:bg-baby-600 transition-colors"
            >
              <Users className="w-4 h-4 mr-2" />
              Add Baby Profile
            </Link>
          </div>
        )}
      </div>

      {/* System Information */}
      {systemStatus && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">System Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Model Type:</span>
                <span className="font-medium">{systemStatus.model_type || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Classes:</span>
                <span className="font-medium">{systemStatus.classes?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Feature Extractor:</span>
                <span className="font-medium">
                  {systemStatus.feature_extractor_ready ? '✅ Ready' : '❌ Not Ready'}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Context Manager:</span>
                <span className="font-medium">
                  {systemStatus.context_manager_ready ? '✅ Ready' : '❌ Not Ready'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Active Profiles:</span>
                <span className="font-medium">{systemStatus.active_profiles || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Preprocessor:</span>
                <span className="font-medium">
                  {systemStatus.preprocessor_ready ? '✅ Ready' : '❌ Not Ready'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard; 