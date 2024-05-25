from azureml.core import Workspace, Model
from azureml.core.webservice import AksWebservice, AksCompute
from azureml.core.model import InferenceConfig
import sklearn
from azureml.core import Environment

def main():
    # Define the environment
    env = Environment("deploytocloudenv")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.python.conda_dependencies.add_pip_package("numpy")
    env.python.conda_dependencies.add_pip_package("scikit-learn=={}".format(sklearn.__version__))
    env.python.conda_dependencies.add_pip_package("pandas")

    # Load the workspace
    ws = Workspace.from_config()

    # Register the model
    model = Model(ws, name='insurance_model')

    # Define the inference configuration
    inference_config = InferenceConfig(entry_script="score.py", environment=env)

    # Create AKS cluster (only if not already created)
    aks_name = "myakscluster"
    try:
        aks_target = AksCompute(ws, aks_name)
        print(f"Found existing AKS cluster: {aks_name}")
    except:
        print(f"Creating new AKS cluster: {aks_name}")
        aks_config = AksCompute.provisioning_configuration(vm_size="Standard_D2_v2")
        aks_target = AksCompute.create(ws, name=aks_name, provisioning_configuration=aks_config)
        aks_target.wait_for_completion(show_output=True)

    # Define the deployment configuration
    deployment_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    # Deploy the model to AKS
    service = Model.deploy(ws, "myservice", [model], inference_config, deployment_config, deployment_target=aks_target)
    service.wait_for_deployment(show_output=True)
    print(service.state)

if __name__ == "__main__":
    main()
