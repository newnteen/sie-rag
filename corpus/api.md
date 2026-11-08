# API keys

You can create and revoke API keys under Settings > Developer. Keys are scoped to
a workspace, not to a user.

To rotate a key, create a replacement first, deploy it to your clients, then revoke
the old key. There is no automatic overlap window, so two keys can be valid at the
same time while you migrate.

Keys expire after 12 months unless you set a shorter lifetime when you create them.
