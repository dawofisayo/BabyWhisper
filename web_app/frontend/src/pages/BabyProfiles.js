import React, { useState, useEffect } from 'react';
import {
  Baby,
  Plus,
  Edit,
  Calendar,
  Clock,
  Utensils,
  Moon,
  Heart,
  Save,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';

const BabyProfiles = () => {
  const [babies, setBabies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingBaby, setEditingBaby] = useState(null);
  const [newBaby, setNewBaby] = useState({
    name: '',
    age_months: '',
    birth_date: ''
  });

  useEffect(() => {
    loadBabies();
  }, []);

  const loadBabies = async () => {
    try {
      const response = await fetch('/api/babies');
      const data = await response.json();
      setBabies(data);
    } catch (error) {
      console.error('Failed to load babies:', error);
      toast.error('Failed to load baby profiles');
    } finally {
      setLoading(false);
    }
  };

  const handleAddBaby = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/babies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newBaby),
      });

      if (response.ok) {
        toast.success('Baby profile created successfully!');
        setNewBaby({ name: '', age_months: '', birth_date: '' });
        setShowAddForm(false);
        loadBabies();
      } else {
        const error = await response.json();
        throw new Error(error.error || 'Failed to create baby profile');
      }
    } catch (error) {
      console.error('Failed to create baby:', error);
      toast.error(error.message);
    }
  };

  const updateBabyContext = async (babyId, contextType, time = new Date()) => {
    try {
      const updateData = {};
      updateData[`${contextType}_time`] = time.toISOString();

      const response = await fetch(`/api/babies/${babyId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      if (response.ok) {
        toast.success(`${contextType} time updated!`);
        loadBabies();
      } else {
        throw new Error('Failed to update context');
      }
    } catch (error) {
      console.error('Failed to update context:', error);
      toast.error('Failed to update care information');
    }
  };

  const formatTimeAgo = (dateString) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    // Handle negative differences (when the time is slightly in the future due to timing/timezone issues)
    if (diffInHours <= 0) {
      return 'Just now';
    }
    
    if (diffInHours < 1) {
      const minutes = Math.floor(diffInHours * 60);
      if (minutes <= 0) return 'Just now';
      return `${minutes} min ago`;
    } else {
      return `${diffInHours.toFixed(1)}h ago`;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Baby className="w-12 h-12 text-baby-500 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-600">Loading baby profiles...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Baby Profiles
          </h1>
          <p className="text-gray-600">
            Manage baby profiles and track care activities for context-aware analysis
          </p>
        </div>
        <button
          onClick={() => setShowAddForm(true)}
          className="inline-flex items-center px-4 py-2 bg-baby-500 text-white rounded-lg hover:bg-baby-600 transition-colors"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Baby
        </button>
      </div>

      {/* Add Baby Form */}
      {showAddForm && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Add New Baby</h2>
            <button
              onClick={() => setShowAddForm(false)}
              className="p-2 text-gray-500 hover:text-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <form onSubmit={handleAddBaby} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Baby Name *
                </label>
                <input
                  type="text"
                  required
                  value={newBaby.name}
                  onChange={(e) => setNewBaby({ ...newBaby, name: e.target.value })}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-baby-500 focus:border-baby-500"
                  placeholder="Enter baby's name"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Age (months)
                </label>
                <input
                  type="number"
                  min="0"
                  max="36"
                  value={newBaby.age_months}
                  onChange={(e) => setNewBaby({ ...newBaby, age_months: parseInt(e.target.value) || '' })}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-baby-500 focus:border-baby-500"
                  placeholder="Age in months"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Birth Date
                </label>
                <input
                  type="date"
                  value={newBaby.birth_date}
                  onChange={(e) => setNewBaby({ ...newBaby, birth_date: e.target.value })}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-baby-500 focus:border-baby-500"
                />
              </div>
            </div>
            
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="inline-flex items-center px-4 py-2 bg-baby-500 text-white rounded-lg hover:bg-baby-600 transition-colors"
              >
                <Save className="w-4 h-4 mr-2" />
                Create Profile
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Baby Cards */}
      {babies.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {babies.map((baby) => (
            <div key={baby.id} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              {/* Baby Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-3 bg-baby-100 rounded-full">
                    <Baby className="w-6 h-6 text-baby-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{baby.name}</h3>
                    <p className="text-sm text-gray-600">{baby.age_months} months old</p>
                  </div>
                </div>
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <Edit className="w-4 h-4" />
                </button>
              </div>

              {/* Care Activities */}
              <div className="space-y-3">
                {/* Feeding */}
                <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Utensils className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-700">Feeding</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-blue-600">
                      {formatTimeAgo(baby.last_feeding)}
                    </span>
                    <button
                      onClick={() => updateBabyContext(baby.id, 'feeding')}
                      className="px-2 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors"
                    >
                      Fed Now
                    </button>
                  </div>
                </div>

                {/* Sleep */}
                <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Moon className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-medium text-purple-700">Sleep</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-purple-600">
                      {formatTimeAgo(baby.last_sleep)}
                    </span>
                    <button
                      onClick={() => updateBabyContext(baby.id, 'sleep')}
                      className="px-2 py-1 bg-purple-500 text-white text-xs rounded hover:bg-purple-600 transition-colors"
                    >
                      Woke Up
                    </button>
                  </div>
                </div>

                {/* Diaper */}
                <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Heart className="w-4 h-4 text-green-600" />
                    <span className="text-sm font-medium text-green-700">Diaper</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-green-600">
                      {formatTimeAgo(baby.last_diaper_change)}
                    </span>
                    <button
                      onClick={() => updateBabyContext(baby.id, 'diaper')}
                      className="px-2 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600 transition-colors"
                    >
                      Changed
                    </button>
                  </div>
                </div>
              </div>

              {/* Birth Date */}
              {baby.birth_date && (
                <div className="mt-4 pt-3 border-t border-gray-200">
                  <div className="flex items-center text-xs text-gray-500">
                    <Calendar className="w-3 h-3 mr-1" />
                    Born: {new Date(baby.birth_date).toLocaleDateString()}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Baby className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Baby Profiles</h3>
          <p className="text-gray-600 mb-6">
            Create your first baby profile to start using context-aware cry analysis
          </p>
          <button
            onClick={() => setShowAddForm(true)}
            className="inline-flex items-center px-6 py-3 bg-baby-500 text-white rounded-lg hover:bg-baby-600 transition-colors"
          >
            <Plus className="w-5 h-5 mr-2" />
            Add Your First Baby
          </button>
        </div>
      )}
    </div>
  );
};

export default BabyProfiles; 