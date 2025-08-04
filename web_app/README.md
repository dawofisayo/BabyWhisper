# 🍼 BabyWhisper Web Dashboard

Modern, responsive web interface for the BabyWhisper AI baby cry classification system.

## ✨ Features

- **🎵 Real-time Cry Analysis** - Upload files or record live audio
- **👶 Baby Profile Management** - Context-aware analysis with care tracking
- **📊 Interactive Dashboard** - System metrics and insights
- **📱 Mobile Responsive** - Works perfectly on all devices
- **🧠 Smart Insights** - AI predictions with intelligent explanations

## 🏗️ Architecture

```
Frontend (React.js)  ←→  Backend (Flask API)  ←→  BabyWhisper AI Core
     Port 3000              Port 5000              (Python ML Models)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with BabyWhisper dependencies installed
- **Node.js 16+** and npm
- **Modern web browser** with microphone access

### 1. Start the Backend API

```bash
# From project root
cd web_app/backend

# Install Flask dependencies
pip install -r requirements.txt

# Start the Flask API server
python app.py
```

The API will be available at `http://localhost:5000`

### 2. Start the Frontend

```bash
# In a new terminal
cd web_app/frontend

# Install React dependencies
npm install

# Start the development server
npm start
```

The web app will open at `http://localhost:3000`

## 📁 Project Structure

```
web_app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── README.md
├── frontend/
│   ├── public/
│   │   └── index.html      # Main HTML template
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   │   └── Navbar.js
│   │   ├── pages/          # Main application pages
│   │   │   ├── Dashboard.js
│   │   │   ├── AudioClassifier.js
│   │   │   ├── BabyProfiles.js
│   │   │   └── Analytics.js
│   │   ├── App.js          # Main React application
│   │   └── index.js        # React entry point
│   ├── package.json        # Node.js dependencies
│   └── README.md
└── README.md               # This file
```

## 🎯 API Endpoints

The Flask backend provides these REST API endpoints:

- `GET /api/health` - System health check
- `GET /api/system-status` - BabyWhisper system status
- `GET /api/babies` - Get all baby profiles
- `POST /api/babies` - Create new baby profile
- `PUT /api/babies/{id}` - Update baby context
- `GET /api/babies/{id}/insights` - Get baby insights
- `POST /api/classify-audio` - Classify uploaded audio
- `POST /api/feedback` - Provide learning feedback

## 🌟 Key Components

### Dashboard
- System overview and quick actions
- Baby profiles preview
- Performance metrics

### Audio Classifier
- File upload with drag & drop
- Live audio recording
- Real-time analysis results
- Context-aware predictions

### Baby Profiles
- Create and manage baby profiles
- Update feeding, sleep, and diaper times
- Track care history

### Analytics
- Model performance metrics
- System specifications
- Technical insights

## 🔧 Development

### Backend Development

```bash
# Run Flask in debug mode
cd web_app/backend
export FLASK_ENV=development
python app.py
```

### Frontend Development

```bash
# Start React development server
cd web_app/frontend
npm start

# Build for production
npm run build
```

### Adding New Features

1. **Backend**: Add new routes to `app.py`
2. **Frontend**: Create components in `src/components/` or pages in `src/pages/`
3. **Styling**: Use Tailwind CSS classes for consistent design

## 🎨 Design System

- **Colors**: Custom baby theme with warm, parent-friendly colors
- **Typography**: Clean, readable fonts optimized for tired parents
- **Icons**: Lucide React for consistent iconography
- **Layout**: Responsive grid system with mobile-first approach

## 🚀 Deployment

### Production Backend
```bash
# Install Gunicorn
pip install gunicorn

# Run production server
cd web_app/backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Production Frontend
```bash
# Build optimized bundle
cd web_app/frontend
npm run build

# Serve with any static file server
npx serve -s build
```

## 🔒 Security Notes

- CORS is enabled for development
- No authentication implemented yet
- File uploads are temporarily stored
- Baby profiles stored in memory (use database for production)

## 🤝 Contributing

1. Follow the existing code structure
2. Use consistent naming conventions
3. Add proper error handling
4. Test both frontend and backend changes
5. Ensure mobile responsiveness

## 📝 License

Part of the BabyWhisper project - MIT License

---

**BabyWhisper Web Dashboard: Beautiful interface for intelligent baby care** 🍼✨ 