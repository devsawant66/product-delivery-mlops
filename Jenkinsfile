pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
    steps {
        bat 'py -m pip install -r requirements.txt'
    }
}

        stage('Train Model') {
            steps {
                bat 'python src/train.py'
            }
        }

        stage('Quality Gate') {
            steps {
                script {
                    def metrics = readJSON file: 'metrics.json'
                    if (metrics.model_mae > 10) {
                        error("Model MAE too high! Failing build.")
                    }
                }
            }
        }

        stage('Start API') {
            steps {
                echo 'Model validated successfully.'
            }
        }
    }
}