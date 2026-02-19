set -euo pipefail

pip install pycocotools pycocoevalcap==1.2

until java --version > /dev/null; do
    # stop unattended-upgrades to avoid conflicts
    sudo systemctl stop unattended-upgrades.service || true
    sudo systemctl disable unattended-upgrades.service || true

    # kill all unattended-upgrade processes (无匹配时 grep 返回 1，需 || true 避免 pipefail 退出)
    ps -ef | grep -i unattended | grep -v 'grep' | awk '{print "sudo kill -9 " $2}' | sh || true

    sudo systemctl stop unattended-upgrades.service || true
    sudo systemctl disable unattended-upgrades.service || true

    sudo rm -rf /tmp/*tpu* || true
    sudo rm -rf /tmp/wandb || true

    # kill all apt processes
    ps -ef | grep apt | awk '{print "sudo kill -9 " $2}' | sh || true

    sleep 5

    timeout 20 sudo apt update -y || true
    timeout 20 sudo apt install -y openjdk-11-jdk || true
done

pip install "jax[tpu]==0.6.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jaxlib==0.6.2
pip install orbax-checkpoint==0.11.32
pip install jaxtyping==0.3.7
pip install sentencepiece webdataset==1.0.2

pip uninstall protobuf -y
pip install protobuf==3.20.3

echo 补 > ~/sqa冲

