pipeline {
  agent any
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
    }
  }
  environment {
    something = '13'
    RELEASE = '69.420'
  }
}