import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import datetime as dt

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Helper Function to Clean Barcodes (Standardization) ---
def clean_barcode(series):
    """
    Forces barcodes to be strings without decimal places.
    Handles floats (885.0 -> '885') and strings (' 885 ' -> '885').
    """
    # 1. Force to numeric, turning errors to NaN
    s = pd.to_numeric(series, errors='coerce')
    # 2. Fill NaNs with 0 temporarily to allow integer conversion, then convert to Int64
    s = s.fillna(0).astype('Int64')
    # 3. Convert to string
    s = s.astype(str)
    # 4. Replace the '0' we added (which represents bad barcodes) with a placeholder or keep as '0'
    # We will just strip whitespace to be safe
    return s.str.strip()

# --- Helper Function to Load Main Sales Data ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded sales XLSX file."""

    # The 26 columns from your sales file, in order.
    SALES_COLUMN_NAMES = [
        "No", "Branch", "Customer/Debtor", "Sales Invoice No", "Sale Date", 
        "Sale Time", "Salesperson", "Product Barcode", "Product Code", 
        "Product Name", "Unit", "Quantity Sold", "Unit Price Before Discount", 
        "Discount per Unit", "Unit Price After Discount", "Cost per Unit", 
        "Profit per Unit", "Total Before Discount", "Total Discount", 
        "Total After Discount", "Total Cost", "Total Profit", "Profit %", 
        "Markup %", "Added By", "Status"
    ]
    
    # Map from your names to the names the script expects
    SALES_COLUMNS_RENAME_MAP = {
        "Sale Date": "Sales Date",
        "Sale Time": "Sales Time",
        "Sales Invoice No": "Invoice No",
        "Total Before Discount": "Total Price Before Discount",
        "Total After Discount": "Total Price After Discount"
    }

    try:
        # 1. Read the Excel file (first sheet, first row is header)
        df = pd.read_excel(uploaded_file, sheet_name=0, header=0)
        
        # 2. Critical Check: Ensure we have AT LEAST the expected columns
        if len(df.columns) < len(SALES_COLUMN_NAMES):
            st.error(f"Error: The uploaded sales file has only {len(df.columns)} columns, but the dashboard expects at least {len(SALES_COLUMN_NAMES)}.")
            return None
            
        # 3. Slice the df to only include the columns we want
        df = df.iloc[:, :len(SALES_COLUMN_NAMES)]
        df.columns = SALES_COLUMN_NAMES
        df.rename(columns=SALES_COLUMNS_RENAME_MAP, inplace=True)

        # --- Date/Time Parsing ---
        df['Sales Time'] = df['Sales Time'].astype(str)
        df['datetime'] = pd.to_datetime(
            df['Sales Date'].astype(str).str.split(' ').str[0] + ' ' + df['Sales Time'], 
            errors='coerce',
        )
        df.dropna(subset=['datetime'], inplace=True)
        if df.empty:
            st.error("Error: All date parsing failed.")
            return None
        
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour_of_day'] = df['datetime'].dt.hour
        
        # 4. Convert numeric columns
        numeric_cols = [
            'Quantity Sold', 'Unit Price Before Discount', 'Discount per Unit',
            'Unit Price After Discount', 'Cost per Unit', 'Profit per Unit',
            'Total Price Before Discount', 'Total Discount', 'Total Price After Discount',
            'Total Cost', 'Total Profit'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Handle Invoice No
        if 'Invoice No' not in df.columns:
            df['Invoice No'] = df.index
        
        # 6. --- FIX: Robust Barcode Cleaning ---
        if 'Product Barcode' not in df.columns:
            df['Product Barcode'] = df['Product Code'].fillna('NO_BARCODE')
        
        # Apply the standardized cleaning function to the sales barcodes
        df['Product Barcode'] = clean_barcode(df['Product Barcode'])
            
        df.dropna(subset=['Total Profit', 'Quantity Sold', 'Invoice No'], inplace=True)

        # 7. Select required columns
        required_cols = [
            'datetime', 'Product Name', 'Quantity Sold', 'Total Price After Discount',
            'Total Cost', 'Total Profit', 'Product Barcode', 'Invoice No',
            'day_of_week', 'hour_of_day'
        ]
        
        # Add placeholders if missing
        for col in required_cols:
            if col not in df.columns:
                if col in ['Quantity Sold', 'Total Price After Discount', 'Total Cost', 'Total Profit']:
                    df[col] = 0
                else:
                    df[col] = 'N/A'
        
        return df[required_cols]

    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return None

# --- Helper Function to Load UTC Data ---
@st.cache_data
def load_utc_data(uploaded_file):
    """Loads and preprocesses the uploaded UTC (CSV) file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'Timestamp' not in df.columns:
            st.error("UTC Data Error: 'Timestamp' column not found.")
            return None
        df['datetime'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        
        UTC_COLUMNS_RENAME_MAP = {
            "Product": "Product Name",
            "Amount Sold": "Quantity Sold",
            "Total Revenue": "Total Price After Discount",
            "Total Cost": "Total Cost",
            "Total Profit": "Total Profit"
        }
        df.rename(columns=UTC_COLUMNS_RENAME_MAP, inplace=True)
        
        if 'Product Name' not in df.columns:
            st.error("UTC Data Error: 'Product' column not found.")
            return None
        
        if 'Amount Gifted' in df.columns:
            df['Amount Gifted'] = pd.to_numeric(df['Amount Gifted'], errors='coerce').fillna(0)
            df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce').fillna(0)
            df['Quantity Sold'] = df['Quantity Sold'] + df['Amount Gifted']
        
        # Create a unique barcode for UTC items (Prefix with UTC_ so they don't mix with Stock)
        df['Product Barcode'] = "UTC_" + df['Product Name'].str.replace(' ', '_')
        
        df['Invoice No'] = [f"UTC_INV_{i}" for i in range(len(df))]
        df.dropna(subset=['datetime'], inplace=True)
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour_of_day'] = df['datetime'].dt.hour
        
        numeric_cols = ['Quantity Sold', 'Total Price After Discount', 'Total Cost', 'Total Profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0
        
        df.dropna(subset=['Total Profit', 'Quantity Sold', 'Invoice No', 'datetime'], inplace=True)
        
        required_cols = [
            'datetime', 'Product Name', 'Quantity Sold', 'Total Price After Discount',
            'Total Cost', 'Total Profit', 'Product Barcode', 'Invoice No',
            'day_of_week', 'hour_of_day'
        ]
        
        return df[required_cols]

    except Exception as e:
        st.error(f"Error loading UTC data: {e}")
        return None

@st.cache_data
def load_stock_data(stock_file):
    """Loads and processes the uploaded stock XLSX file."""

    STOCK_COLUMN_NAMES = [
        "No.", "Branch Name", "Main Supplier", "Product Code", "Barcode", 
        "Product Name", "Stock", "Average Cost", "Selling Price 1", "Profit", 
        "Profit %", "Total Sales (1)", "Total Cost", "Total Profit", "Unit", 
        "Account Period Cost", "Latest Purchase Cost", "Total Stock (All Units)", 
        "Generic Drug Name", "Storage", "Product Type", "Product Group", "Brand", 
        "Selling Price 2", "Selling Price 3", "Selling Price 4", "Selling Price 5", 
        "Tax Exempt Product", "Decimal Places", "Status", "Name 2", "Name 3"
    ]
    
    try:
        df = pd.read_excel(stock_file, sheet_name=0, header=0)

        if len(df.columns) < len(STOCK_COLUMN_NAMES):
            st.error(f"Error: The uploaded stock file has only {len(df.columns)} columns.")
            return None
            
        df = df.iloc[:, :len(STOCK_COLUMN_NAMES)]
        df.columns = STOCK_COLUMN_NAMES
        
        if 'Barcode' not in df.columns or 'Stock' not in df.columns or 'Product Name' not in df.columns or 'Average Cost' not in df.columns:
            st.error("Stock file is missing required columns.")
            return None
        
        df_stock = df[['Barcode', 'Stock', 'Product Name', 'Average Cost']].copy()
        df_stock['Stock'] = pd.to_numeric(df_stock['Stock'], errors='coerce')
        df_stock['Average Cost'] = pd.to_numeric(df_stock['Average Cost'], errors='coerce')

        # --- FIX: Use exact same cleaning logic as Sales Data ---
        # This converts "885...0" floats into "885..." strings
        df_stock['Barcode'] = clean_barcode(df_stock['Barcode'])
        
        # Drop rows where data is bad (excluding stock, as stock can be 0)
        df_stock.dropna(subset=['Barcode', 'Average Cost'], inplace=True)
        
        # Aggregate duplicates (same barcode listed twice)
        df_stock_agg = df_stock.groupby('Barcode').agg(
            Stock=('Stock', 'sum'),
            product_name=('Product Name', 'first'),
            Average_Cost=('Average Cost', 'mean')
        ).reset_index()
        
        df_stock_agg['total_value'] = df_stock_agg['Stock'] * df_stock_agg['Average_Cost']
        
        return df_stock_agg

    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None
        
@st.cache_data
def get_transaction_kpis(df):
    unique_invoices = df['Invoice No'].nunique()
    if unique_invoices == 0:
        return 0, 0, 0
    total_revenue = df['Total Price After Discount'].sum()
    total_items = df['Quantity Sold'].sum()
    aov = total_revenue / unique_invoices
    items_per_transaction = total_items / unique_invoices
    return aov, items_per_transaction, unique_invoices

@st.cache_data
def get_common_co_purchases(df, selected_product, top_n=5):
    target_invoices = df[df['Product Name'] == selected_product]['Invoice No'].unique()
    if len(target_invoices) == 0:
        return pd.DataFrame(columns=['Product Name', 'count'])
    co_purchase_df = df[df['Invoice No'].isin(target_invoices)]
    other_items_df = co_purchase_df[co_purchase_df['Product Name'] != selected_product]
    if other_items_df.empty:
        return pd.DataFrame(columns=['Product Name', 'count'])
    top_items = other_items_df['Product Name'].value_counts().head(top_n).reset_index()
    top_items.columns = ['Product Name', 'count']
    return top_items


# --- Main App ---
st.title("🛒 Sales & Inventory Dashboard")

# --- Sidebar for File Upload ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("1. Upload Sales Data (.xlsx)", type=["xlsx"])
utc_file = st.sidebar.file_uploader("3. Upload UTC Data (.csv)", type=["csv"])
stock_file = st.sidebar.file_uploader("2. Upload Stock Data (.xlsx)", type=["xlsx"])

# --- Load and Process Data ---
df_main = None
df_utc = None
all_dataframes = []

if uploaded_file is not None:
    df_main = load_data(uploaded_file)
    if df_main is not None:
        all_dataframes.append(df_main)

if utc_file is not None:
    df_utc = load_utc_data(utc_file)
    if df_utc is not None:
        all_dataframes.append(df_utc)

# --- Load stock data ---
df_stock = None
if stock_file is not None:
    df_stock = load_stock_data(stock_file) 


if not all_dataframes:
    st.info("Please upload your sales data file (and optionally UTC data) using the sidebar to get started.")
else:
    # --- Combine all loaded dataframes ---
    df_full = pd.concat(all_dataframes, ignore_index=True)
    
    if 'growth_factor' not in st.session_state:
        st.session_state.growth_factor = 1.0

    # --- Date Range Filter ---
    st.sidebar.header("Date Range")
    min_date = df_full['datetime'].min().date()
    max_date = df_full['datetime'].max().date()

    selected_dates = st.sidebar.date_input(
        "Filter by Date:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    mask = (df_full['datetime'].dt.date >= start_date) & (df_full['datetime'].dt.date <= end_date)
    df = df_full.loc[mask].copy()

    # --- Calculate num_days based on filtered data ---
    num_days = df['datetime'].dt.date.nunique()
    if num_days == 0:
        num_days = 1 

    st.sidebar.info(f"Showing data from: **{start_date}** to **{end_date}**")
    st.sidebar.success(f"Calculating velocity based on **{num_days}** unique sales day(s).")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Product Filters")
    all_products = df['Product Name'].unique()
    excluded_products = st.sidebar.multiselect(
        "Select products to EXCLUDE from dashboard:", 
        options=all_products,
        default=[] 
    )
    
    df_filtered = df[~df['Product Name'].isin(excluded_products)] 
    
    st.sidebar.header("Stock Configuration")
    lead_time_weeks = st.sidebar.number_input(
        "Weeks of stock to keep on hand?", 
        min_value=1, max_value=12, value=3, step=1
    )
    
    if df_filtered.empty:
        st.warning("No data found for the selected date range and filters.")
    else:
        tab_list = [
            "📈 Overall Trend & Forecasting",  
            "🔥 Busiest Times Heatmap",      
            "🏆 Product Performance", 
            "🛒 Customer Purchase Insights", 
            "📦 Inventory Insights"
        ]
        
        if stock_file is not None:
            tab_list.append("🛍️ Reorder & Stock Check")
        
        tabs = st.tabs(tab_list)

        # --- Tab 1: Overall Trend & Forecasting ---
        with tabs[0]:
            st.header("Overall Sales Trend & Forecasting")

            st.subheader("Period-Wide KPIs")
            st.markdown(f"These metrics reflect the *filtered dataset* ({start_date} to {end_date}).")
            
            total_revenue = df_filtered['Total Price After Discount'].sum()
            total_profit = df_filtered['Total Profit'].sum()
            total_cost_of_goods_sold = df_filtered['Total Cost'].sum()

            df_daily_stats = df_filtered.groupby(df_filtered['datetime'].dt.date) \
                                      .agg(
                                          daily_sales=('Total Price After Discount', 'sum'),
                                          daily_profit=('Total Profit', 'sum') 
                                      ) \
                                      .reset_index()
            df_daily_stats = df_daily_stats.rename(columns={'datetime': 'date'})
            
            aov, items_per_tx, total_tx = get_transaction_kpis(df_filtered)

            avg_daily_sales = total_revenue / num_days if num_days > 0 else 0
            profit_margin_pct = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
            avg_daily_transactions = total_tx / num_days if num_days > 0 else 0

            total_inventory_value = 0
            if df_stock is not None:
                if 'total_value' in df_stock.columns:
                    total_inventory_value = df_stock['total_value'].sum()
                else:
                    st.warning("Could not calculate Inventory Value.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Revenue", f"฿{total_revenue:,.2f}")
            col2.metric("Total Profit", f"฿{total_profit:,.2f}")
            col3.metric("Cost of Goods Sold (COGS)", f"฿{total_cost_of_goods_sold:,.2f}")

            if df_stock is not None:
                col4, col5, col6 = st.columns(3)
                col4.metric("Current Inventory Value", f"฿{total_inventory_value:,.2f}")
                col5.metric("Average Order Value (AOV)", f"฿{aov:,.2f}")
                col6.metric("Avg. Items per Transaction", f"{items_per_tx:,.2f}")

            else:
                col4.metric("Total Transactions", f"{total_tx:,}")
                col5.metric("Average Order Value (AOV)", f"฿{aov:,.2f}")
                col6.metric("Avg. Items per Transaction", f"{items_per_tx:,.2f}")
            
            col7, col8, col9 = st.columns(3)
            col7.metric("Average Daily Sales", f"฿{avg_daily_sales:,.2f}")
            col8.metric("Profit Margin", f"{profit_margin_pct:,.2f}%")
            col9.metric("Avg. Daily Transactions", f"{avg_daily_transactions:,.2f}")
            
            st.markdown("---")

            st.subheader("Overall Daily Sales & Profit Trend")
            df_melted = df_daily_stats.melt(
                id_vars=['date'], 
                value_vars=['daily_sales', 'daily_profit'], 
                var_name='Metric', 
                value_name='Amount (฿)'
            )
            df_melted['Metric'] = df_melted['Metric'].map({
                'daily_sales': 'Total Sales',
                'daily_profit': 'Total Profit'
            })

            fig_trend = px.line(
                df_melted,
                x='date',
                y='Amount (฿)',
                color='Metric', 
                title="Total Daily Sales & Profit Over Time",
                labels={'date': 'Date', 'Amount (฿)': 'Amount (฿)'},
                color_discrete_map={'Total Sales': 'blue', 'Total Profit': 'green'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            df_daily_transactions = df_filtered.groupby(df_filtered['datetime'].dt.date) \
                                              .agg(daily_transactions=('Invoice No', 'nunique')) \
                                              .reset_index()
            df_daily_transactions = df_daily_transactions.rename(columns={'datetime': 'date'})

            fig_bar_transactions = px.bar(
                df_daily_transactions,
                x='date',
                y='daily_transactions',
                title="Total Transactions Per Day",
                labels={'date': 'Date', 'daily_transactions': 'Total Transactions'}
            )
            st.plotly_chart(fig_bar_transactions, use_container_width=True)


            # --- Prophet Forecasting ---
            st.subheader("Sales Forecast with Prophet")
            df_prophet_revenue = df_daily_stats[['date', 'daily_sales']].rename(
                columns={'date': 'ds', 'daily_sales': 'y'}
            )
            
            if len(df_prophet_revenue) < 5:
                st.warning("Not enough daily data to generate a forecast.")
            else:
                forecast_days = st.slider(
                    "Days to forecast into the future", 
                    21, 365, 30,
                    key="forecast_days_slider"
                )
                
                @st.cache_data
                def get_prophet_forecast(data, periods):
                    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                    m.fit(data)
                    future = m.make_future_dataframe(periods=periods)
                    forecast = m.predict(future)
                    return m, forecast

                with st.spinner(f"Generating {forecast_days}-day forecast..."):
                    try:
                        m_revenue, forecast_revenue = get_prophet_forecast(df_prophet_revenue, forecast_days)
                        
                        try:
                            last_actual_date = df_prophet_revenue['ds'].max()
                            last_actual_date_dt = pd.to_datetime(last_actual_date)
                            future_date_dt = last_actual_date_dt + pd.Timedelta(days=28)
                            
                            recent_pred_series = forecast_revenue.loc[forecast_revenue['ds'] == last_actual_date_dt, 'yhat']
                            future_pred_series = forecast_revenue.loc[forecast_revenue['ds'] == future_date_dt, 'yhat']

                            if recent_pred_series.empty or future_pred_series.empty:
                                st.session_state.growth_factor = 1.0
                            else:
                                recent_pred = recent_pred_series.values[0]
                                future_pred = future_pred_series.values[0]
                                if recent_pred > 0 and future_pred > 0:
                                    st.session_state.growth_factor = future_pred / recent_pred
                                else:
                                    st.session_state.growth_factor = 1.0 
                        except Exception as e:
                            st.session_state.growth_factor = 1.0
                        
                        st.subheader(f"{forecast_days}-Day Sales (Revenue) Forecast")
                        fig_forecast = plot_plotly(m_revenue, forecast_revenue)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        st.subheader("Revenue Forecast Components")
                        fig_components = m_revenue.plot_components(forecast_revenue)
                        st.pyplot(fig_components)
                    
                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")

                st.subheader("Forecasted Growth Factor")
                st.metric("Predicted 4-Week Sales Growth (from trend)", f"{st.session_state.growth_factor:.2%}")

        # --- Tab 2: Busiest Times Heatmap ---
        with tabs[1]:
            st.header("Busiest Times Heatmap")
            
            df_heatmap_base = df_filtered[
                (df_filtered['hour_of_day'] >= 9) & (df_filtered['hour_of_day'] <= 20)
            ]

            if df_heatmap_base.empty:
                st.warning("No sales data found between 9 AM and 8 PM.")
            else:
                heatmap_data = df_heatmap_base.groupby(['day_of_week', 'hour_of_day'])['Invoice No'].nunique().reset_index()
                
                heatmap_pivot = heatmap_data.pivot(
                    index='day_of_week',
                    columns='hour_of_day',
                    values='Invoice No'
                ).fillna(0)
                
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                hour_order = list(range(9, 21))
                heatmap_pivot = heatmap_pivot.reindex(index=day_order, columns=hour_order, fill_value=0)
                
                fig = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Transactions"),
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Daily Summary Statistics (9am-8pm)")
                
                try:
                    daily_agg = df_heatmap_base.groupby(df_heatmap_base['datetime'].dt.date).agg(
                        daily_revenue=('Total Price After Discount', 'sum'),
                        daily_transactions=('Invoice No', 'nunique')
                    )
                    daily_agg['day_of_week'] = pd.to_datetime(daily_agg.index).day_name()
                    
                    day_summary = daily_agg.groupby('day_of_week').agg(
                        avg_revenue=('daily_revenue', 'mean'),
                        avg_transactions=('daily_transactions', 'mean')
                    )
                    
                    busiest_hours = heatmap_pivot.idxmax(axis=1)
                    quietest_hours = heatmap_pivot.idxmin(axis=1)
                    
                    day_summary = day_summary.join(busiest_hours.rename('Busiest Hour (Transactions)'))
                    day_summary = day_summary.join(quietest_hours.rename('Quietest Hour (Transactions)'))
                    day_summary = day_summary.reindex(day_order)
                    
                    st.dataframe(
                        day_summary,
                        use_container_width=True,
                        column_config={
                            "avg_revenue": st.column_config.NumberColumn("Avg. Revenue", format="฿%.2f"),
                            "avg_transactions": st.column_config.NumberColumn("Avg. Transactions", format="%.1f")
                        }
                    )
                except Exception:
                    pass

        # --- Tab 3: Product Performance ---
        with tabs[2]:
            st.header("Product Performance")
            st.markdown(f"Analyzed over **{num_days}** days of sales.")
            
            product_performance = df_filtered.groupby('Product Name').agg(
                total_quantity_sold=('Quantity Sold', 'sum'),
                total_profit=('Total Profit', 'sum'),
                total_revenue=('Total Price After Discount', 'sum')
            ).reset_index()
            
            st.subheader("Top Performing Products")
            sort_by = st.selectbox("Sort top products by:", ["Total Profit", "Total Quantity Sold", "Total Revenue"], index=0)
            
            if sort_by == "Total Profit":
                sorted_products = product_performance.sort_values(by='total_profit', ascending=False)
                chart_col = 'total_profit'
            elif sort_by == "Total Quantity Sold":
                sorted_products = product_performance.sort_values(by='total_quantity_sold', ascending=False)
                chart_col = 'total_quantity_sold'
            else:
                sorted_products = product_performance.sort_values(by='total_revenue', ascending=False)
                chart_col = 'total_revenue'

            col1, col2 = st.columns(2)
            if not sorted_products.empty:
                best_product = sorted_products.iloc[0]
                col1.metric(f"Best Product (by {sort_by})", best_product['Product Name'])
                col2.metric(f"Value (by {sort_by})", f"฿{best_product[chart_col]:,.2f}" if chart_col != 'total_quantity_sold' else f"{best_product[chart_col]:,.0f} Units")

            st.subheader("Top 10 Products")
            top_10_products = sorted_products.head(10)
            
            if not top_10_products.empty:
                fig_bar = px.bar(
                    top_10_products,
                    x='Product Name',
                    y=chart_col,
                    title=f"Top 10 Products by {sort_by}",
                    text_auto=True
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with st.expander("View All Product Performance Data"):
                st.dataframe(sorted_products, use_container_width=True)

        # --- Tab 4: Customer Purchase Insights ---
        with tabs[3]:
            st.header("🛒 Customer Purchase Insights")
            
            aov, items_per_tx, total_tx = get_transaction_kpis(df_filtered)
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Order Value (AOV)", f"฿{aov:,.2f}")
            col2.metric("Avg. Items per Transaction", f"{items_per_tx:,.2f}")
            col3.metric("Total Transactions", f"{total_tx:,}")

            st.markdown("---")
            st.subheader("What products are bought together?")
            
            top_product_list = df_filtered['Product Name'].value_counts().head(200).index.tolist()
            
            if top_product_list:
                selected_product = st.selectbox("Select a product:", options=top_product_list)
                if selected_product:
                    co_purchase_data = get_common_co_purchases(df_filtered, selected_product, top_n=5)
                    if not co_purchase_data.empty:
                        fig_co = px.bar(
                            co_purchase_data,
                            x='count',
                            y='Product Name',
                            orientation='h',
                            title=f"Top 5 Products Bought With '{selected_product}'"
                        )
                        fig_co.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_co, use_container_width=True)

        # --- Tab 5: Inventory Insights ---
        with tabs[4]:
            st.header("Inventory Insights & Sales Velocity")
            growth_factor = st.session_state.growth_factor
            
            group_cols = ['Product Name', 'Product Barcode']
            inventory_stats = df_filtered.groupby(group_cols).agg(
                total_quantity_sold=('Quantity Sold', 'sum'),
            ).reset_index()
            
            inventory_stats.rename(columns={'Product Barcode': 'Barcode'}, inplace=True)
            # The Barcodes are already cleaned in load_data, but we ensure string type for display
            inventory_stats['Barcode'] = inventory_stats['Barcode'].astype(str)
            
            inventory_stats['avg_daily_sales'] = inventory_stats['total_quantity_sold'] / num_days
            inventory_stats['avg_weekly_sales'] = inventory_stats['avg_daily_sales'] * 7
            
            st.subheader("Stocking Suggestions")
            st.markdown(f"`Suggested Stock Level` is based on **{lead_time_weeks} weeks** of safety stock (adj. by {growth_factor:.2%} growth).")
            
            inventory_stats['suggested_stock_level'] = (inventory_stats['avg_weekly_sales'] * lead_time_weeks) * growth_factor
            inventory_stats['suggested_stock_level'] = np.ceil(inventory_stats['suggested_stock_level'])
            
            st.dataframe(
                inventory_stats.sort_values(by='avg_weekly_sales', ascending=False),
                use_container_width=True,
                column_config={
                    "Barcode": st.column_config.TextColumn("Barcode"),
                    "avg_weekly_sales": st.column_config.NumberColumn("Avg. Weekly Sales", format="%.2f"),
                    "suggested_stock_level": st.column_config.NumberColumn(f"Suggested {lead_time_weeks}-Week Stock", format="%.0f units"),
                }
            )
        
        # --- Tab 6: Reorder & Stock Check ---
        if stock_file is not None:
            with tabs[5]:
                st.header("Stock vs. Forecasted Demand")
                growth_factor = st.session_state.growth_factor
                
                if df_stock is not None:
                    # Filter out UTC items for Restock analysis
                    df_restock_sales_data = df_filtered[
                        ~df_filtered['Product Barcode'].astype(str).str.startswith('UTC_')
                    ]
                    
                    if df_restock_sales_data.empty:
                        st.warning("No non-UTC sales data found to calculate restock levels.")
                    else:
                        group_cols = ['Product Name', 'Product Barcode']
                        inventory_stats_restock = df_restock_sales_data.groupby(group_cols).agg(
                            total_quantity_sold=('Quantity Sold', 'sum'),
                        ).reset_index()

                        inventory_stats_restock.rename(columns={'Product Barcode': 'Barcode'}, inplace=True)
                        # Barcodes are already cleaned from load_data
                        
                        inventory_stats_restock['avg_daily_sales'] = inventory_stats_restock['total_quantity_sold'] / num_days
                        inventory_stats_restock['avg_weekly_sales'] = inventory_stats_restock['avg_daily_sales'] * 7
                        inventory_stats_restock['suggested_stock_level'] = np.ceil((inventory_stats_restock['avg_weekly_sales'] * lead_time_weeks) * growth_factor)
                        
                        # --- MERGE ---
                        # Because we used clean_barcode() on both, the join keys should now match
                        df_merged = pd.merge(
                            inventory_stats_restock,
                            df_stock[['Barcode', 'Stock']], 
                            on='Barcode',
                            how='left' 
                        )
                        
                        df_merged['Stock'] = df_merged['Stock'].fillna(0)
                        df_merged['deficit'] = df_merged['suggested_stock_level'] - df_merged['Stock']
                        df_merged['deficit'] = np.ceil(df_merged['deficit'])
                        
                        st.subheader("Reorder List")
                        
                        df_reorder = df_merged[df_merged['deficit'] > 0].sort_values(by='deficit', ascending=False)
                        all_reorder_products = df_reorder['Product Name'].unique()
                        
                        if 'ordered_list' not in st.session_state:
                            st.session_state.ordered_list = []

                        st.session_state.ordered_list = [p for p in st.session_state.ordered_list if p in all_reorder_products]

                        ordered_items = st.multiselect(
                            "Mark items as 'Already Ordered' (to hide from list):",
                            options=all_reorder_products,
                            key='ordered_list'
                        )
                        
                        df_reorder_to_display = df_reorder[~df_reorder['Product Name'].isin(ordered_items)]
                        
                        st.dataframe(
                            df_reorder_to_display, 
                            use_container_width=True,
                            column_config={
                                "Product Name": st.column_config.TextColumn("Product Name (from Sales)", width="large"),
                                "Barcode": st.column_config.TextColumn("Barcode"),
                                "Stock": st.column_config.NumberColumn("Current Stock", format="%.0f units"),
                                "suggested_stock_level": st.column_config.NumberColumn("Suggested Stock (Adj.)", format="%.0f units"),
                                "deficit": st.column_config.NumberColumn("Need to Order", format="%.0f units"),
                            }
                        )
                        
                        with st.expander("View Full Stock-Sales Comparison (All Non-UTC Items)"):
                            st.dataframe(df_merged.sort_values(by='deficit', ascending=False), use_container_width=True)
        
        elif uploaded_file is not None or utc_file is not None:
            st.sidebar.warning("Upload your stock file to enable the 'Reorder & Stock Check' tab.")
