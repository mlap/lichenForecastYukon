# Function to install virtual env with requisite packages
installPyVirtualEnv <- function(venv_name) {
  envs <- virtualenv_list()
  if (!(venv_name %in% envs)) {
    message("Creating virtual environment: ", venv_name)
    virtualenv_create(envname = venv_name)
    
    # Names of required packages
    req_packages <- c("PyDrive", "earthengine-api", "numpy", "pandas", "scikit-learn", "geemap", "xgboost")
    
    # Install required packages into venv
    virtualenv_install(envname = venv_name, packages = req_packages, ignore_installed = TRUE)
  }
  
  # Use this virtual env in reticulate
  use_virtualenv(venv_name, required = TRUE)
  
  py_config()
  
}