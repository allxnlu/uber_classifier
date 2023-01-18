pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                bat '''
                python --version
                echo "hello im fine"
                '''
            }
        }
        stage('test') {
            steps {
                bat '''
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