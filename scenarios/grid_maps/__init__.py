"""Compatibility package for the engine's legacy import path.

``core/user_agents/red/grid_deploy_agent.py`` imports
``scenarios.grid_maps.deployment``; the plan validator itself lives in the
top-level ``grid_maps`` package.  This thin shim keeps ``core/`` untouched.
"""
