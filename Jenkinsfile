pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                bat '''
                python --version
                echo "hello im fine"
                pip install virtualenv
                virtualenv venv 
                venv\\Scripts\\activate
                '''
            }
        }
        stage('test') {
            steps {
                bat '''
                pip install -r requirements.txt
                python chatbot.py
                '''
            }
        }
    }
}