import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Brain,
  Activity,
  Clock,
  Baby,
  Zap,
  Target
} from 'lucide-react';

const Analytics = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [babies, setBabies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalyticsData();
  }, []);

  const loadAnalyticsData = async () => {
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
      console.error('Failed to load analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 text-baby-500 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  const systemMetrics = [
    {
      label: 'Model Accuracy',
      value: '83.7%',
      icon: Target,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      description: 'Accuracy on real Donate-a-Cry dataset'
    },
    {
      label: 'Audio Features',
      value: '293',
      icon: Zap,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      description: 'Optimized audio characteristics analyzed'
    },
    {
      label: 'Active Profiles',
      value: babies.length.toString(),
      icon: Baby,
      color: 'text-baby-600',
      bgColor: 'bg-baby-50',
      description: 'Baby profiles for context-aware analysis'
    },
    {
      label: 'Model Type',
      value: 'Ensemble',
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      description: 'Random Forest + SVM + MLP combination'
    }
  ];

  const modelPerformance = [
    { model: 'Random Forest', accuracy: 82.1, color: 'bg-green-500' },
    { model: 'SVM', accuracy: 84.2, color: 'bg-blue-500' },
    { model: 'MLP', accuracy: 85.1, color: 'bg-purple-500' },
    { model: 'Ensemble', accuracy: 83.7, color: 'bg-baby-500' }
  ];

  const cryCategories = [
    { category: 'Hunger', color: 'bg-red-400' },
    { category: 'Pain', color: 'bg-orange-400' },
    { category: 'Discomfort', color: 'bg-yellow-400' },
    { category: 'Tiredness', color: 'bg-blue-400' },
    { category: 'Burping', color: 'bg-green-400' }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Analytics Dashboard
        </h1>
        <p className="text-gray-600">
          System performance metrics and insights from BabyWhisper AI
        </p>
      </div>

      {/* System Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemMetrics.map((metric, index) => {
          const IconComponent = metric.icon;
          return (
            <div
              key={index}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-3">
                <div className={`p-3 rounded-lg ${metric.bgColor}`}>
                  <IconComponent className={`w-6 h-6 ${metric.color}`} />
                </div>
                <TrendingUp className="w-4 h-4 text-gray-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 mb-1">
                  {metric.value}
                </p>
                <p className="text-sm font-medium text-gray-600 mb-2">
                  {metric.label}
                </p>
                <p className="text-xs text-gray-500">
                  {metric.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Model Performance */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <Brain className="w-6 h-6 mr-2 text-baby-600" />
            Model Performance
          </h2>
          <div className="text-sm text-gray-500">
            Accuracy on real Donate-a-Cry dataset (457 recordings)
          </div>
        </div>

        <div className="space-y-4">
          {modelPerformance.map((model, index) => (
            <div key={index} className="flex items-center space-x-4">
              <div className="w-24 text-sm font-medium text-gray-700">
                {model.model}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-sm text-gray-600">Accuracy</div>
                  <div className="text-sm font-semibold text-gray-900">
                    {model.accuracy}%
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${model.color}`}
                    style={{ width: `${model.accuracy}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Technical Specifications */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Activity className="w-6 h-6 mr-2 text-blue-600" />
            Technical Specifications
          </h2>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-600">Audio Processing</span>
              <span className="font-medium">323 features per sample</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-600">Sampling Rate</span>
              <span className="font-medium">22,050 Hz</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-600">Audio Duration</span>
              <span className="font-medium">3 seconds optimal</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-600">Feature Types</span>
              <span className="font-medium">MFCC, Spectral, Temporal</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-600">Training Dataset</span>
              <span className="font-medium">457 real baby recordings</span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-gray-600">Context Factors</span>
              <span className="font-medium">Age, Feeding, Sleep, Diaper</span>
            </div>
          </div>
        </div>

        {/* Cry Categories */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Target className="w-6 h-6 mr-2 text-green-600" />
            Cry Categories
          </h2>
          
          <div className="space-y-3">
            {cryCategories.map((category, index) => (
              <div key={index} className="flex items-center space-x-3">
                <div className={`w-4 h-4 rounded ${category.color}`} />
                <span className="text-gray-700 font-medium">{category.category}</span>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-2">Classification Process</h3>
            <div className="text-sm text-gray-600 space-y-1">
              <p>1. Audio preprocessing & feature extraction</p>
              <p>2. Ensemble model prediction</p>
              <p>3. Context-aware adjustment</p>
              <p>4. Smart insights generation</p>
            </div>
          </div>
        </div>
      </div>

      {/* Baby Profiles Summary */}
      {babies.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Baby className="w-6 h-6 mr-2 text-baby-600" />
            Baby Profiles Summary
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {babies.slice(0, 3).map((baby) => (
              <div key={baby.id} className="p-4 bg-baby-50 rounded-lg border border-baby-100">
                <h3 className="font-medium text-baby-900 mb-2">{baby.name}</h3>
                <div className="space-y-1 text-sm text-baby-700">
                  <div className="flex justify-between">
                    <span>Age:</span>
                    <span>{baby.age_months} months</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Last Fed:</span>
                    <span>
                      {baby.last_feeding 
                        ? new Date(baby.last_feeding).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
                        : 'N/A'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Last Sleep:</span>
                    <span>
                      {baby.last_sleep 
                        ? new Date(baby.last_sleep).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
                        : 'N/A'
                      }
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Future Enhancements */}
      <div className="bg-gradient-to-r from-baby-50 to-blue-50 rounded-xl border border-baby-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Clock className="w-6 h-6 mr-2 text-baby-600" />
          Coming Soon
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Enhanced Analytics</h3>
            <ul className="space-y-1 text-gray-600">
              <li>• Historical cry pattern analysis</li>
              <li>• Feeding/sleep correlation insights</li>
              <li>• Personalized baby development tracking</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Advanced Features</h3>
            <ul className="space-y-1 text-gray-600">
              <li>• Real-time monitoring dashboard</li>
              <li>• Predictive cry analysis</li>
              <li>• Parent care optimization suggestions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics; 