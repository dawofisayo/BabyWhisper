import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Baby, 
  Mic, 
  Users, 
  BarChart3, 
  Activity,
  Menu,
  X
} from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const location = useLocation();

  useEffect(() => {
    // Check system status on load
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to check system status:', error);
    }
  };

  const navItems = [
    { path: '/', label: 'Dashboard', icon: BarChart3 },
    { path: '/classify', label: 'Analyze Cry', icon: Mic },
    { path: '/babies', label: 'Baby Profiles', icon: Users },
    { path: '/analytics', label: 'Analytics', icon: Activity },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="p-2 bg-baby-500 rounded-lg group-hover:bg-baby-600 transition-colors">
              <Baby className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">BabyWhisper</h1>
              <p className="text-xs text-gray-500">AI Baby Care Assistant</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const IconComponent = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive(item.path)
                      ? 'bg-baby-50 text-baby-700 border border-baby-200'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <IconComponent className="w-4 h-4 mr-2" />
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* System Status & Mobile Menu */}
          <div className="flex items-center space-x-3">
            {/* System Status Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  systemStatus?.baby_whisper_ready ? 'bg-green-400' : 'bg-red-400'
                }`}
              />
              <span className="text-xs text-gray-500">
                {systemStatus?.baby_whisper_ready ? 'Online' : 'Offline'}
              </span>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100"
            >
              {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden py-4 border-t border-gray-200">
            <div className="space-y-1">
              {navItems.map((item) => {
                const IconComponent = item.icon;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setIsOpen(false)}
                    className={`flex items-center px-4 py-3 rounded-lg text-sm font-medium ${
                      isActive(item.path)
                        ? 'bg-baby-50 text-baby-700'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    }`}
                  >
                    <IconComponent className="w-4 h-4 mr-3" />
                    {item.label}
                  </Link>
                );
              })}
              
              {/* Mobile system status */}
              <div className="flex items-center px-4 py-3 text-sm">
                <div
                  className={`w-2 h-2 rounded-full mr-3 ${
                    systemStatus?.baby_whisper_ready ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                <span className="text-gray-500">
                  System: {systemStatus?.baby_whisper_ready ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 