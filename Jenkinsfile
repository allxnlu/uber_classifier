pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh '''
                python --version
                echo "hello im fine"
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