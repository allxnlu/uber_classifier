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
                pip install -U virtualenv
                virtualenv venv 
                . venv/bin/activate
                pip install -r requirements.txt
                python chatbot.py
                '''
            }
        }
    }
}