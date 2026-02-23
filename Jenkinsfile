pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
    steps {
        bat '"C:\\Users\\User\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" -m pip install -r requirements.txt --no-cache-dir'
    }
}

        stage('Train Model') {
            steps {
                bat '"C:\\Users\\User\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" -u src/train.py'
            }
        }

        stage('Show Metrics') {
            steps {
                bat 'type metrics.json'
            }
        }

    }
}