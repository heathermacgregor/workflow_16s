# ... (keep all the imports and other functions unchanged) ...

from string import Template

def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20,
    cfg: Optional[Dict] = None
) -> None:
    if cfg:
        group_col = cfg.get("group_column", "nuclear_contamination_status")
        group_col_values = cfg.get("group_column_values", [True, False])
    else:
        group_col = "nuclear_contamination_status"
        group_col_values = [True, False]
        
    figures_dict = _extract_figures(amplicon_data)
    
    include_sections = include_sections or [
        k for k, v in figures_dict.items() if v
    ]
    if 'violin' in figures_dict and 'violin' not in include_sections:
        include_sections.append('violin')
    
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Top features tables (without SHAP data)
    x = 1
    group_1_name = f"{group_col}={group_col_values[x-1]}"
    group_1_df = _prepare_features_table(
        getattr(amplicon_data, f'top_features_group_{x}', []),
        max_features,
        group_1_name
    )
    x = 2
    group_2_name = f"{group_col}={group_col_values[x-1]}"
    group_2_df = _prepare_features_table(
        getattr(amplicon_data, f'top_features_group_{x}', []),
        max_features,
        group_2_name
    )
    
    # Stats summary
    stats_df = _prepare_stats_summary(
        amplicon_data.stats
    )
    
    # ML summary (includes SHAP reports)
    ml_metrics, ml_features, shap_reports = _prepare_ml_summary(
        amplicon_data.models,
        amplicon_data.top_features_group_1,
        amplicon_data.top_features_group_2
    )
    ml_html = _format_ml_section(ml_metrics, ml_features, shap_reports) if ml_metrics is not None else "<p>No ML results available</p>"
    
    tables_html = f"""
    <div class="subsection">
        <h3>Top Features</h3>
        <h4>Features associated with {group_1_name}</h4>
        {_add_table_functionality(group_1_df, 'contam-table')}
        
        <h4>Features Associated with {group_2_name}</h4>
        {_add_table_functionality(group_2_df, 'pristine-table')}
    </div>
    
    <div class="subsection">
        <h3>Statistical Summary</h3>
        {_add_table_functionality(stats_df, 'stats-table')}
    </div>
    
    <div class="subsection">
        <h3>Machine Learning Results</h3>
        {ml_html}
    </div>
    """

    id_counter = itertools.count()
    sections, plot_data = _prepare_sections(
        figures_dict, include_sections, id_counter
    )
    sections_html = "\n".join(_section_html(s) for s in sections)

    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )

    # Prepare payload with proper JavaScript escaping
    payload = json.dumps(plot_data, cls=NumpySafeJSONEncoder, ensure_ascii=False)
    payload = payload.replace("\\", "\\\\").replace("'", "\\'").replace('</', '<\\/')

    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
    
    try:
        css_content = css_path.read_text(encoding='utf-8') 
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = ""
        
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

    # Use custom delimiters to avoid conflicts with curly braces
    class CustomTemplate(Template):
        delimiter = '@@'
        idpattern = r'[a-z][_a-z0-9]*'
    
    # Create a safe substitution mapping
    safe_substitutes = {
        'title': "16S Amplicon Analysis Report",
        'plotly_js_tag': plotly_js_tag,
        'generated_ts': ts,
        'section_list': ", ".join(include_sections),
        'nav_html': nav_html,
        'tables_html': tables_html,
        'sections_html': sections_html,
        'plot_data_json': payload,
        'table_js': table_js,
        'css_content': css_content
    }
    
    # Create template and perform safe substitution
    template = CustomTemplate(html_template)
    html = template.safe_substitute(safe_substitutes)
        
    output_path.write_text(html, encoding="utf-8")
