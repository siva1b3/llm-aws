output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_identity" {
  value = data.aws_caller_identity.current.arn
}

output "desired_machine_states" {
  description = "Preset target for every configured machine, including those to destroy."
  value       = local.desired_states
}

output "machines" {
  description = "Created instances and their connection addresses."
  value = {
    for name, instance in aws_instance.machine : name => {
      id         = instance.id
      public_ip  = instance.public_ip
      private_ip = instance.private_ip
      is_spot    = local.machine_controls[name].is_spot
      state      = aws_ec2_instance_state.machine[name].state
    }
  }
}
