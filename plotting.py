#!/usr/bin/env python3
"""
Player Statistics Visualizer (Bright Colors + Brackets)
========================================================
Generates scatter plots, distributions, box plots, and fine-grained skill brackets.
Saves outputs as .png files with highly saturated, bright colors.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "squadDB.sqlite"

def load_data():
    print("Loading data from DB...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database {DB_PATH} not found.")
        
    conn = sqlite3.connect(DB_PATH)
    
    try:
        general_df = pd.read_sql_query("SELECT player_id, games_played, general_skill, sigma FROM player_ratings", conn)
        role_df = pd.read_sql_query("SELECT player_id, role, role_skill, games_in_role FROM player_role_ratings", conn)
    except Exception as e:
        print(f"Error loading tables. Did you run the pipeline first? {e}")
        conn.close()
        return None, None
        
    conn.close()
    
    # Clean data
    general_df = general_df[general_df['games_played'] > 0]
    role_df = role_df[role_df['games_in_role'] > 0]
    role_df = role_df[role_df['role'] != 'Unknown']
    
    return general_df, role_df

def plot_scatter_grid(general_df, role_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Skill vs. Matches Played', fontsize=18, fontweight='bold')
    
    # 1. General Skill vs Matches Played (Bright Blue)
    sns.scatterplot(data=general_df, x='games_played', y='general_skill', alpha=0.4, s=20, ax=axes[0, 0], color='dodgerblue')
    axes[0, 0].set_title('General Skill vs. Total Matches Played')
    axes[0, 0].set_xlabel('Total Matches Played')
    axes[0, 0].set_ylabel('General Skill (Logits)')
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)

    # 2. Role Skill vs Matches Played in Role
    top_roles = role_df['role'].value_counts().nlargest(8).index
    plot_df = role_df[role_df['role'].isin(top_roles)]
    
    sns.scatterplot(data=plot_df, x='games_in_role', y='role_skill', hue='role', palette='bright', alpha=0.5, s=20, ax=axes[0, 1])
    axes[0, 1].set_title('Role Skill vs. Matches Played in Role (Top 8 Roles)')
    axes[0, 1].set_xlabel('Matches Played in Specific Role')
    axes[0, 1].set_ylabel('Role Skill (Logits)')
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].legend(title='Role', fontsize=8)

    # 3. Uncertainty (Sigma) vs Matches Played (Bright Red)
    sns.scatterplot(data=general_df, x='games_played', y='sigma', alpha=0.4, s=20, ax=axes[1, 0], color='crimson')
    axes[1, 0].set_title('Rating Uncertainty (Sigma) vs. Matches Played')
    axes[1, 0].set_xlabel('Total Matches Played')
    axes[1, 0].set_ylabel('Sigma (Uncertainty)')
    axes[1, 0].set_xscale('log')

    # 4. General Skill vs Role Skill
    merged_df = role_df.merge(general_df[['player_id', 'general_skill']], on='player_id')
    merged_df = merged_df[merged_df['role'].isin(top_roles)]
    
    sns.scatterplot(data=merged_df, x='general_skill', y='role_skill', hue='role', palette='bright', alpha=0.4, s=20, ax=axes[1, 1])
    axes[1, 1].set_title('General Skill vs. Role Skill')
    axes[1, 1].set_xlabel('General Skill')
    axes[1, 1].set_ylabel('Role Skill')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot_1_scatter_grid.png', dpi=150)
    plt.close()
    print("   Saved: plot_1_scatter_grid.png")

def plot_distributions(general_df, role_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Skill Distributions', fontsize=18, fontweight='bold')
    
    # 1. Overall General Skill Distribution (Bright Purple)
    sns.histplot(general_df['general_skill'], bins=50, kde=True, ax=axes[0], color='darkviolet')
    axes[0].set_title('Distribution of General Skill')
    axes[0].set_xlabel('General Skill (Logits)')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Role Skill Distribution by Role (Boxplot) using 'hls' for max brightness
    role_counts = role_df['role'].value_counts()
    valid_roles = role_counts[role_counts >= 500].index
    plot_df = role_df[role_df['role'].isin(valid_roles)]
    
    sns.boxplot(data=plot_df, x='role', y='role_skill', hue='role', legend=False, ax=axes[1], 
                order=valid_roles, palette='hls')
    axes[1].set_title('Role Skill Distribution by Role')
    axes[1].set_xlabel('Role')
    axes[1].set_ylabel('Role Skill (Logits)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot_2_distributions.png', dpi=150)
    plt.close()
    print("   Saved: plot_2_distributions.png")

def plot_specialization_and_popularity(general_df, role_df):
    merged_df = role_df.merge(general_df[['player_id', 'general_skill']], on='player_id')
    merged_df['specialization_bonus'] = merged_df['role_skill'] - merged_df['general_skill']
    
    role_counts = role_df['role'].value_counts()
    valid_roles = role_counts[role_counts >= 500].index
    plot_df = merged_df[merged_df['role'].isin(valid_roles)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Role Analysis', fontsize=18, fontweight='bold')
    
    # 1. Specialization Bonus using 'hls'
    sns.boxplot(data=plot_df, x='role', y='specialization_bonus', hue='role', legend=False, ax=axes[0], 
                order=valid_roles, palette='hls')
    axes[0].set_title('Specialization Bonus by Role\n(How much better are they at this role vs their general skill?)')
    axes[0].set_xlabel('Role')
    axes[0].set_ylabel('Role Skill - General Skill (Logits)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Role Popularity using 'hls'
    role_pop = role_df.groupby('role')['games_in_role'].sum().sort_values(ascending=False).reset_index()
    role_pop = role_pop[role_pop['role'].isin(valid_roles)]
    
    sns.barplot(data=role_pop, x='role', y='games_in_role', hue='role', legend=False, ax=axes[1], palette='hls')
    axes[1].set_title('Total Matches Played by Role')
    axes[1].set_xlabel('Role')
    axes[1].set_ylabel('Total Matches in Role')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plot_3_specialization_popularity.png', dpi=150)
    plt.close()
    print("   Saved: plot_3_specialization_popularity.png")

def plot_skill_brackets(general_df, role_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Fine-Grained Skill Brackets (Bins of 0.001)', fontsize=18, fontweight='bold')
    
    # 1. General Skill Brackets
    # edgecolor='none' prevents the bars from turning into a giant black block
    sns.histplot(general_df['general_skill'], binwidth=0.001, binrange=(-0.04, 0.04), ax=axes[0], color='dodgerblue', edgecolor='none')
    axes[0].set_title('General Skill Bracket Distribution')
    axes[0].set_xlabel('General Skill (Logits)')
    axes[0].set_ylabel('Player Count')
    axes[0].set_xlim(-0.04, 0.04)
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.7)

    # 2. Role Skill Brackets
    sns.histplot(role_df['role_skill'], binwidth=0.001, binrange=(-0.04, 0.04), ax=axes[1], color='crimson', edgecolor='none')
    axes[1].set_title('Role Skill Bracket Distribution')
    axes[1].set_xlabel('Role Skill (Logits)')
    axes[1].set_ylabel('Player-Role Count')
    axes[1].set_xlim(-0.04, 0.04)
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plot_4_skill_brackets.png', dpi=150)
    plt.close()
    print("   Saved: plot_4_skill_brackets.png")

def main():
    general_df, role_df = load_data()
    if general_df is None:
        return
        
    print("Generating and saving plots...")
    # Darkgrid makes bright colors pop much better
    sns.set_theme(style="darkgrid", palette="bright")
    
    plot_scatter_grid(general_df, role_df)
    plot_distributions(general_df, role_df)
    plot_specialization_and_popularity(general_df, role_df)
    plot_skill_brackets(general_df, role_df)
    
    print("\nDone! Check the current directory for the .png files.")

if __name__ == "__main__":
    main()