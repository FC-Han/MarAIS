from API import GFW

token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJGaXNoaW5nIGVmZm9ydCIsInVzZXJJZCI6MjQ4MTgsImFwcGxpY2F0aW9uTmFtZSI6IkZpc2hpbmcgZWZmb3J0IiwiaWQiOjE3MDUsInR5cGUiOiJ1c2VyLWFwcGxpY2F0aW9uIn0sImlhdCI6MTcyMTczMzk5NSwiZXhwIjoyMDM3MDkzOTk1LCJhdWQiOiJnZnciLCJpc3MiOiJnZncifQ.mNUcrS-5dfbztCgryH3dfBFx-pi01v-7PW7tT8PjEsyhs_KjqTLdJrwpOtLjCPxn9_t8MBqfelEUNeit5u4mYkMKpJu717Exy0sh32DGfhlZN6pXNeUBiEIxLTVhYv23t-FTxnthIN8HYDiuBiix13ULsS7iQQw5mRkpYhElzR93eIydsXpzDO2lWJNom6IHXKChzc8JQgeeqDZdANivuQ1VW_L-bCDv3FRQ8AHIjEtFHhp6UI0uMZdEL0ZouANmpzyNLnM8NYrs1Pl9DeJ3vcmMNLrOu8WiL0M9swmzTTByV-6hl5S6wYeKaDCUIskRCth_Bxfz1qzUCKmHUMPxo6swVGAHatT43wKD2W35W4xfaE1C8Zu6FsOVQE3-3TokXrN86MqpZZulxc-GaBXWgAOQgOUPRnZuSTIdjgcTomiiqNsXy7naNf-pbS81T57YrTZ_IWUo8ExDk_80aNtel4n7YYzXECDR86KJ4YGNVZK51aqopFUMwum49pOWZP9i'

gfw = GFW(token=token)

print(gfw.get_region_id("ALB",  "EEZ"))
