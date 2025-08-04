#!/usr/bin/env python3
"""Test runner for BabyWhisper project."""

import unittest
import sys
import os
import time
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_all_tests():
    """Run all tests and return results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests and capture results
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time


def generate_test_report(result, duration):
    """Generate a comprehensive test report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': round(duration, 2),
        'tests_run': result.testsRun,
        'tests_failed': len(result.failures),
        'tests_errored': len(result.errors),
        'tests_skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': round((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100, 2) if result.testsRun > 0 else 0,
        'failures': [{'test': str(failure[0]), 'message': failure[1]} for failure in result.failures],
        'errors': [{'test': str(error[0]), 'message': error[1]} for error in result.errors]
    }
    
    return report


def print_test_summary(report):
    """Print a formatted test summary."""
    print("\n" + "="*60)
    print("🧪 BABYWHISPER TEST SUITE RESULTS")
    print("="*60)
    print(f"📅 Timestamp: {report['timestamp']}")
    print(f"⏱️  Duration: {report['duration_seconds']} seconds")
    print(f"📊 Tests Run: {report['tests_run']}")
    print(f"✅ Success Rate: {report['success_rate']}%")
    print(f"❌ Failures: {report['tests_failed']}")
    print(f"⚠️  Errors: {report['tests_errored']}")
    print(f"⏭️  Skipped: {report['tests_skipped']}")
    
    if report['failures']:
        print("\n🔴 FAILURES:")
        for failure in report['failures']:
            print(f"   • {failure['test']}")
            print(f"     {failure['message'][:100]}...")
    
    if report['errors']:
        print("\n🟡 ERRORS:")
        for error in report['errors']:
            print(f"   • {error['test']}")
            print(f"     {error['message'][:100]}...")
    
    print("\n" + "="*60)


def save_test_report(report, filename='test_report.json'):
    """Save test report to file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Test report saved to: {filename}")


def run_specific_test_category(category):
    """Run tests for a specific category."""
    categories = {
        'audio': 'test_audio_processing.py',
        'models': 'test_models.py',
        'context': 'test_context.py',
        'web': 'test_web_app.py',
        'integration': 'test_integration.py'
    }
    
    if category not in categories:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return
    
    test_file = categories[category]
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return
    
    print(f"🧪 Running {category} tests...")
    
    # Load and run specific test file
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern=test_file)
    
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    report = generate_test_report(result, end_time - start_time)
    print_test_summary(report)
    
    return result


def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        run_specific_test_category(category)
    else:
        print("🧪 Running all BabyWhisper tests...")
        print("💡 Tip: Run specific test categories with:")
        print("   python tests/run_tests.py audio")
        print("   python tests/run_tests.py models")
        print("   python tests/run_tests.py context")
        print("   python tests/run_tests.py web")
        print("   python tests/run_tests.py integration")
        print()
        
        result, duration = run_all_tests()
        report = generate_test_report(result, duration)
        print_test_summary(report)
        save_test_report(report)


if __name__ == '__main__':
    main() 