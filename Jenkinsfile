pipeline {
  agent { docker {image 'python 3.10.7'}}
  
  stages {
    stage('PRINT') {
      steps {
        echo "This is the build no ${BUILD_NUMBER} and something is ${something}"
      }
    }
    stage("CONFIRM"){
        input{
          message 'Do you confirm this build?'
          ok 'YES'
          parameters {
            string(name: 'TARGET_ENVIRONMENT', defaultValue: 'CONF', description: 'confirmation environment')
          }
        }
        steps{
          echo "Deploying release ${RELEASE} to environment ${TARGET_ENVIRONMENT}"
          writeFile file: 'deployed_release.txt', text: 'the build is deployed and released'
        }
    }

  }
  post{
    success{
      echo 'luv u like a love song baby.'
      archiveArtifacts 'deployed_release.txt'
      bat '''
        dir
        py helloworld.py
      '''
      // python --version
    }
  }
  environment {
    something = '13'
    RELEASE = '69.420'
    PATH = 'C:\\Windows\\System32'
  }
}