

ansible -i inventory.ini ec2 -m ping


ansible -i inventory.ini ec2 -a "sudo apt update"


ansible-playbook -i inventory.ini apt-update.yaml

ansible-playbook -i inventory.ini packages.yaml






ansible -i inventory.ini ec2 -b -m ansible.builtin.apt -a "update_cache=yes"


