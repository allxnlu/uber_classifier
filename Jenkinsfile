pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh '''
                python3 --version
                echo "hello im fine"
                '''
            }
        }
        stage('test') {
            steps {
                sh '''
                pip install virtualenv
                virtualenv venv
                venv\Scripts\activate
                pip install -r requirements.txt
                python chatbot.py
                '''
            }
        }
    }
}