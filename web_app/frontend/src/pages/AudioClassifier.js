import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  Upload,
  Play,
  Pause,
  Square,
  Loader,
  AlertCircle,
  CheckCircle,
  Baby,
  Clock,
  Brain,
  Heart,
  Lightbulb
} from 'lucide-react';
import toast from 'react-hot-toast';

const AudioClassifier = () => {
  const [babies, setBabies] = useState([]);
  const [selectedBaby, setSelectedBaby] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);
  const audioPlayerRef = useRef(null);

  useEffect(() => {
    loadBabies();
    return () => {
      // Cleanup
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    };
  }, []);

  const loadBabies = async () => {
    try {
      const response = await fetch('/api/babies');
      const data = await response.json();
      setBabies(data);
    } catch (error) {
      console.error('Failed to load babies:', error);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedBlob(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      toast.success('Recording started');
    } catch (error) {
      console.error('Failed to start recording:', error);
      toast.error('Failed to access microphone');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
      toast.success('Recording stopped');
    }
  };

  const playRecording = () => {
    if (recordedBlob && !isPlaying) {
      const audioUrl = URL.createObjectURL(recordedBlob);
      audioPlayerRef.current = new Audio(audioUrl);
      
      audioPlayerRef.current.onended = () => {
        setIsPlaying(false);
      };
      
      audioPlayerRef.current.play();
      setIsPlaying(true);
    } else if (audioPlayerRef.current && isPlaying) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('audio/')) {
        setAudioFile(file);
        setRecordedBlob(null);
        setAnalysisResult(null);
        toast.success('Audio file selected');
      } else {
        toast.error('Please select an audio file');
      }
    }
  };

  const analyzeAudio = async () => {
    const audioToAnalyze = audioFile || recordedBlob;
    if (!audioToAnalyze) {
      toast.error('Please upload an audio file or record a cry');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioToAnalyze, 'baby_cry.wav');
      
      if (selectedBaby) {
        formData.append('baby_id', selectedBaby);
      }

      const response = await fetch('/api/classify-audio', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (response.ok) {
        setAnalysisResult(result);
        toast.success('Analysis complete!');
      } else {
        throw new Error(result.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error(`Analysis failed: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBg = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-50 border-green-200';
    if (confidence >= 0.6) return 'bg-yellow-50 border-yellow-200';
    return 'bg-red-50 border-red-200';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Baby Cry Analysis
        </h1>
        <p className="text-gray-600">
          Upload an audio file or record a cry for AI-powered analysis with context-aware insights
        </p>
      </div>

      {/* Baby Selection */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Select Baby Profile (Optional)
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Choose a baby profile for context-aware analysis that considers feeding, sleep, and care history.
        </p>
        <select
          value={selectedBaby}
          onChange={(e) => setSelectedBaby(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-baby-500 focus:border-baby-500"
        >
          <option value="">No baby profile (basic analysis only)</option>
          {babies.map((baby) => (
            <option key={baby.id} value={baby.id}>
              {baby.name} ({baby.age_months} months)
            </option>
          ))}
        </select>
      </div>

      {/* Audio Input Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* File Upload */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Upload className="w-5 h-5 mr-2" />
            Upload Audio File
          </h2>
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-baby-400 transition-colors">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-2">
                Drag and drop or click to select
              </p>
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
                id="audio-upload"
              />
              <label
                htmlFor="audio-upload"
                className="inline-flex items-center px-4 py-2 bg-baby-500 text-white rounded-lg hover:bg-baby-600 cursor-pointer transition-colors"
              >
                Choose File
              </label>
            </div>
            {audioFile && (
              <div className="p-3 bg-baby-50 rounded-lg">
                <p className="text-sm font-medium text-baby-700">
                  Selected: {audioFile.name}
                </p>
                <p className="text-xs text-baby-600">
                  Size: {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Recording */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Mic className="w-5 h-5 mr-2" />
            Record Live Audio
          </h2>
          <div className="space-y-4">
            <div className="text-center py-6">
              {isRecording && (
                <div className="flex items-center justify-center mb-4">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse mr-2"></div>
                  <span className="text-red-600 font-medium">
                    Recording: {formatTime(recordingTime)}
                  </span>
                </div>
              )}
              
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                  isRecording
                    ? 'bg-red-500 hover:bg-red-600'
                    : 'bg-baby-500 hover:bg-baby-600'
                }`}
              >
                {isRecording ? (
                  <Square className="w-6 h-6 text-white" />
                ) : (
                  <Mic className="w-6 h-6 text-white" />
                )}
              </button>
              
              <p className="text-xs text-gray-600 mt-2">
                {isRecording ? 'Click to stop recording' : 'Click to start recording'}
              </p>
            </div>

            {recordedBlob && (
              <div className="p-3 bg-green-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-green-700">
                    Recording ready ({formatTime(recordingTime)})
                  </span>
                  <button
                    onClick={playRecording}
                    className="flex items-center px-3 py-1 bg-green-500 text-white rounded text-xs hover:bg-green-600 transition-colors"
                  >
                    {isPlaying ? <Pause className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />}
                    {isPlaying ? 'Pause' : 'Play'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Analyze Button */}
      <div className="text-center">
        <button
          onClick={analyzeAudio}
          disabled={isAnalyzing || (!audioFile && !recordedBlob)}
          className="inline-flex items-center px-8 py-4 bg-baby-500 text-white rounded-xl font-semibold hover:bg-baby-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all text-lg"
        >
          {isAnalyzing ? (
            <>
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Brain className="w-5 h-5 mr-2" />
              Analyze Baby Cry
            </>
          )}
        </button>
      </div>

      {/* Analysis Results */}
      {analysisResult && !analysisResult.error && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
            <CheckCircle className="w-6 h-6 mr-2 text-green-500" />
            Analysis Results
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Main Result */}
            <div className={`p-6 rounded-xl border-2 ${getConfidenceBg(analysisResult.confidence)}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Primary Classification</h3>
                <Baby className="w-6 h-6 text-baby-600" />
              </div>
              
              <div className="text-center mb-4">
                <p className="text-3xl font-bold text-gray-900 mb-2">
                  {analysisResult.prediction?.charAt(0).toUpperCase() + 
                   analysisResult.prediction?.slice(1)}
                </p>
                <p className={`text-lg font-semibold ${getConfidenceColor(analysisResult.confidence)}`}>
                  {(analysisResult.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>

              {analysisResult.baby_used && (
                <div className="flex items-center justify-center text-sm text-gray-600">
                  <Heart className="w-4 h-4 mr-1" />
                  Analysis for {analysisResult.baby_used}
                </div>
              )}
            </div>

            {/* Explanation */}
            <div className="p-6 bg-blue-50 rounded-xl border border-blue-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Lightbulb className="w-5 h-5 mr-2 text-blue-600" />
                Smart Insights
              </h3>
              <p className="text-gray-700 leading-relaxed">
                {analysisResult.explanation}
              </p>
            </div>
          </div>

          {/* Detailed Probabilities */}
          {analysisResult.all_probabilities && (
            <div className="mt-6 p-6 bg-gray-50 rounded-xl">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Detailed Probabilities
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(analysisResult.all_probabilities).map(([category, probability]) => (
                  <div key={category} className="text-center">
                    <p className="text-sm font-medium text-gray-600 mb-1 capitalize">
                      {category}
                    </p>
                    <p className="text-lg font-bold text-gray-900">
                      {(probability * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                      <div
                        className="bg-baby-500 h-2 rounded-full transition-all"
                        style={{ width: `${probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
            <div className="mt-6 p-6 bg-green-50 rounded-xl border border-green-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
                Recommendations
              </h3>
              <ul className="space-y-2">
                {analysisResult.recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0" />
                    <span className="text-gray-700">{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Timestamp */}
          <div className="mt-4 text-center">
            <p className="text-xs text-gray-500 flex items-center justify-center">
              <Clock className="w-3 h-3 mr-1" />
              Analyzed at {new Date(analysisResult.timestamp).toLocaleString()}
            </p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {analysisResult?.error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6">
          <div className="flex items-start">
            <AlertCircle className="w-6 h-6 text-red-500 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-red-900 mb-2">
                Analysis Failed
              </h3>
              <p className="text-red-700">
                {analysisResult.error}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioClassifier; 