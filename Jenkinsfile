pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh '''
                python --version
                '''
            }
        }
        stage('test') {
            steps {
                sh '''
                python chatbot.py
                '''
            }
        }
    }
}