pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh '''
                python --version
                python print("hello world")
                '''
            }
        }
    }
}