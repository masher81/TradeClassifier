def live_trading_loop():
    """Optimized main trading loop"""
    positions = {}
    print(f"\n▶️ Starting live trader (SL/TP version) - Dry run={DRY_RUN}")
    
    # Validate cache first
    validate_cache()
    
    # Initial market snapshot
    initial_data = {}
    loaded_symbols = 0
    for sym in SYMBOLS[:5]:  # Just load a few for the snapshot
        df = load_history(sym)
        if df is not None and not df.empty:
            initial_data[sym] = df
            loaded_symbols += 1
    
    if DEBUG_MODE:
        print(f"\n🔍 Initial Load: {loaded_symbols}/{min(5, len(SYMBOLS))} symbols loaded")
    
    print_market_summary(initial_data)
    
    try:
        while True:
            cycle_start = time.time()
            print(f"\n⏳ [{datetime.utcnow()}] Open positions: {len(positions)}")
            
            # Refresh cache every 15 minutes
            if datetime.utcnow().minute % 15 == 0:
                RAW_CACHE.clear()
                PROCESSED_CACHE.clear()
                if VERBOSE:
                    print("♻️ Cache cleared")
            
            # Process symbols in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_symbol, sym, positions) for sym in SYMBOLS]
                for future in as_completed(futures):
                    result = future.result()
                    if DEBUG_MODE and result:
                        print(f"  {result} completed")
            
            # Sleep until next candle
            elapsed = time.time() - cycle_start
            sleep_time = max(60 - elapsed, 1)
            if VERBOSE: 
                print(f"💤 Sleeping {sleep_time:.1f}s until next candle")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped cleanly")
