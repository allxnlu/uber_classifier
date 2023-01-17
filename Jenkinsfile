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
                python helloworld.py
                '''
            }
        }
    }
}