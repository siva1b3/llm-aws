data "aws_ami" "ubuntu_2404" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "ubuntu" {
  ami           = data.aws_ami.ubuntu_2404.id
  instance_type = "t3a.medium"

  key_name = aws_key_pair.ec2_key.key_name

  vpc_security_group_ids = [
    data.aws_security_group.main.id
  ]

  tags = {
    Name = "terraform-ubuntu-24-04"
  }
}

output "ec2_public_ip" {
  value = aws_instance.ubuntu.public_ip
}