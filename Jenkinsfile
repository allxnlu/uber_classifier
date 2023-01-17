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
                virtualenv venv --distribute
                . venv/Scripts/activate
                sudo pip install -r requirements.txt
                python chatbot.py
                '''
            }
        }
    }
}