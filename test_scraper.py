import requests
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import io
import base64
import warnings

def solve():
    """
    Scrapes the Wikipedia page for Nobel laureates, processes the data,
    and answers a series of questions.
    """
    # 1. Download and parse the webpage
    URL = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(URL, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(json.dumps([f"Error: Could not fetch the URL. {e}"]))
        return

    try:
        # 2. Automatically detect and extract relevant tables using pandas
        tables = pd.read_html(io.StringIO(response.text))
    except Exception as e:
        print(json.dumps([f"Error: Failed to parse tables from the page. {e}"]))
        return

    # 3. Dynamically identify the main laureates table
    main_df = None
    for df in tables:
        # Heuristic: The main table is large and has specific columns.
        if df.shape[0] < 50:
            continue
        
        cols = {str(c).lower().replace('prize', '').strip() for c in df.columns}
        # Check for essential columns to confirm it's the right table
        required_cols = {'year', 'physics', 'chemistry', 'literature', 'peace'}
        if required_cols.issubset(cols):
            main_df = df
            break
            
    if main_df is None:
        print(json.dumps(["Error: Could not identify the main Nobel laureates table on the page. Please check the page structure."]))
        return
        
    # Dynamically map column names based on substring matching
    rename_map = {}
    for col in main_df.columns:
        col_str = str(col).lower()
        if 'year' in col_str:
            rename_map[col] = 'Year'
        elif 'physics' in col_str:
            rename_map[col] = 'Physics'
        elif 'chemistry' in col_str:
            rename_map[col] = 'Chemistry'
        elif 'physiology or medicine' in col_str:
            rename_map[col] = 'Physiology or Medicine'
        elif 'literature' in col_str:
            rename_map[col] = 'Literature'
        elif 'peace' in col_str:
            rename_map[col] = 'Peace'
        elif 'economic' in col_str:
            rename_map[col] = 'Economics'
    
    main_df = main_df.rename(columns=rename_map)
    
    category_cols = [c for c in rename_map.values() if c != 'Year']
    
    melted_df = main_df.melt(
        id_vars=['Year'],
        value_vars=category_cols,
        var_name='Category',
        value_name='LaureatesRaw'
    )
    
    # 4. Clean the data
    melted_df['Year'] = melted_df['Year'].astype(str).str.extract(r'(\d{4})', expand=False)
    melted_df.dropna(subset=['Year', 'LaureatesRaw'], inplace=True)
    melted_df = melted_df[~melted_df['LaureatesRaw'].str.lower().str.contains('not awarded', na=False)]
    melted_df['Year'] = pd.to_numeric(melted_df['Year'])

    laureates_data = []
    for _, row in melted_df.iterrows():
        year = row['Year']
        category = row['Category']
        cell_text = str(row['LaureatesRaw'])

        lines = cell_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Clean the line: remove prize share and footnotes
            line = re.sub(r'^\d/\d+:\s*', '', line)
            line = re.sub(r'\[\w+\]', '', line).strip()
            
            # Extract name and country using regex
            country_match = re.search(r'\(([^)]+)\)$', line)
            if country_match:
                country = country_match.group(1).strip()
                name = line[:country_match.start()].strip()
            else:
                country = None
                name = line.strip()
            
            name = name.rstrip(',').strip()
            name = re.sub(r'\s*\(born in.*', '', name, flags=re.IGNORECASE)

            if name:
                laureates_data.append({
                    'Year': int(year),
                    'Category': category,
                    'Laureate': name,
                    'Country': country
                })

    df = pd.DataFrame(laureates_data)

    # --- Answer Questions ---

    # 1. How many Indian Nobel laureates have won in Peace or Literature?
    cat_filter = df['Category'].isin(['Peace', 'Literature'])
    country_filter = df['Country'].str.contains('India', na=False, case=False)
    indian_laureates = df[cat_filter & country_filter]
    num_indian_laureates = indian_laureates['Laureate'].nunique()
    ans1 = f"{num_indian_laureates} Indian Nobel laureates have won in the Peace or Literature categories."

    # 2. Which decade had the highest number of Nobel laureates overall?
    df['Decade'] = (df['Year'] // 10) * 10
    laureates_per_decade = df.groupby('Decade')['Laureate'].count()
    if not laureates_per_decade.empty:
        top_decade = laureates_per_decade.idxmax()
        ans2 = f"The {top_decade}s was the decade with the highest number of Nobel laureates."
    else:
        ans2 = "Could not determine the decade with the highest number of laureates."

    # 3. Identify any laureate who has won more than once and mention the categories.
    laureate_counts = df['Laureate'].value_counts()
    multiple_winners_series = laureate_counts[laureate_counts > 1]
    
    multi_winner_details = []
    for name, count in multiple_winners_series.items():
        categories = df[df['Laureate'] == name]['Category'].tolist()
        multi_winner_details.append(f"{name} ({count} wins): {', '.join(sorted(categories))}")
        
    if not multi_winner_details:
        ans3 = "No laureate appears to have won more than once based on the parsed data."
    else:
        ans3 = "Laureates who have won more than once: " + "; ".join(multi_winner_details)

    # 4. Plot a line chart showing the number of laureates per year from 1901 to 2023.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Suppress Matplotlib font warnings
    
    df_plot = df[(df['Year'] >= 1901) & (df['Year'] <= 2023)].copy()
    laureates_per_year = df_plot.groupby('Year')['Laureate'].count()
    all_years = range(1901, 2024)
    laureates_per_year = laureates_per_year.reindex(all_years, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    ax.plot(laureates_per_year.index, laureates_per_year.values, marker='o', linestyle='-', markersize=3, linewidth=1.5)
    ax.set_title('Number of Nobel Laureates per Year (1901-2023)', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Laureates', fontsize=12)
    ax.set_xlim(1900, 2024)
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    ans4 = f'data:image/png;base64,{img_base64}'
    plt.close(fig)
    
    if len(ans4) > 100000:
        ans4 = "Warning: The generated image is larger than the 100,000 byte limit."

    # --- Final Output ---
    answers = [ans1, ans2, ans3, ans4]
    print(json.dumps(answers))

solve()