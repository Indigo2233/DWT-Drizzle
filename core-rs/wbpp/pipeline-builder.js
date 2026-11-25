// WBPP injection point for DWT-Drizzle.
//
// Select this file as Pipeline builder in WBPP:
//   Pipeline > Pipeline scripting > Pipeline builder
//
// The standard WBPP light pipeline is assembled first. The DWT operation is
// queued before standard integration. Rust can register the calibrated frames.

engine.pipelineManager.buildPipelineForLight();

let dwtOperationIndex = findDwtInsertionIndex();
let dwtOperationBlock = new BPPOperationBlock( "DWT-Drizzle (Rust)", undefined, true );
dwtOperationBlock.envForScript = function()
{
   return { name: this.name, status: this.status };
};
dwtOperationBlock._run = function( env, requestInterruption )
{
   let cfg = readDwtConfiguration( engine );
   console.noteln( "* DWT-Drizzle Pipeline builder: <raw>" + engine.pipelineBuilderScriptFile + "</raw>" );
   console.flush();
   let dwtDirectory = engine.outputDirectory + "/DWT";
   let rasterDirectory = dwtDirectory + "/rasters";
   File.createDirectory( rasterDirectory, true );
   File.createDirectory( dwtDirectory + "/runs", true );

   let groups = engine.groupsManager.groupsForMode( BPP.GroupingMode.POST ).filter( function( group )
   {
      return group.isActive && group.imageType == ImageType.Light
         && group.associatedRGBchannel != BPP.AssociatedChannel.COMBINED_RGB;
   } );
   let completed = 0;
   for ( let g = 0; g < groups.length; ++g )
   {
      let group = groups[ g ];
      if ( group.fastIntegrationData && group.fastIntegrationData.enabled )
      {
         console.warningln( "** Warning: DWT-Drizzle skips FAST Integration group " + group.folderName( false ) + "." );
         continue;
      }
      let activeFrames = group.activeFrames().filter( function( frame )
      {
         return dwtSourceForFrame( frame ) != undefined;
      } );
      if ( activeFrames.length == 0 )
         continue;
      if ( group.isCFA && group.associatedRGBchannel == undefined )
      {
         console.warningln( "** Warning: DWT-Drizzle skips undebayered CFA group " + group.folderName( false ) + "." );
         continue;
      }
      let firstSource = dwtSourceForFrame( activeFrames[ 0 ] );
      let firstWindow = ImageWindow.open( firstSource )[ 0 ];
      if ( firstWindow == null || firstWindow.isNull )
      {
         console.warningln( "** Warning: cannot open calibrated frame: " + firstSource );
         continue;
      }
      let width = firstWindow.mainView.image.width;
      let height = firstWindow.mainView.image.height;
      firstWindow.forceClose();
      let groupName = sanitizeDwtName( group.folderName( false ) );
      dwtOperationBlock.groupDescription = groupName + " | Exporting calibrated frames";
      console.noteln( "* DWT-Drizzle exporting " + activeFrames.length + " calibrated frames for " + groupName );
      console.flush();
      let groupDirectory = dwtDirectory + "/runs/" + groupName;
      File.createDirectory( groupDirectory, true );
      let prepareRequest = makeDwtRequest( engine, groupDirectory, width, height, cfg );
      for ( let f = 0; f < activeFrames.length; ++f )
      {
         dwtOperationBlock.groupDescription = groupName + " | Exporting calibrated frame "
            + ( f + 1 ) + "/" + activeFrames.length;
         CoreApplication.processEvents();
         let frame = activeFrames[ f ];
         let source = dwtSourceForFrame( frame );
         let stem = sanitizeDwtName( File.extractName( source ) );
         let rasterFile = groupDirectory + "/" + f + "-" + stem + ".f32le";
         if ( !writeCalibratedFloat32( source, rasterFile, width, height ) )
         {
            console.warningln( "** Warning: cannot export frame as a matching single-channel raster: " + source );
            continue;
         }
         let descriptor = frame.descriptor || {};
         let processed = frame.fileItem.processed[ frame.associatedID ] || {};
         let xdrz = cfg.useXdrzSeeds && frame.drizzleFile ? frame.drizzleFile : null;
         prepareRequest.frames.push( {
            enabled: true,
            source: source,
            registered: processed[ BPP.FrameProcessingStep.REGISTRATION ] || null,
            xdrz: xdrz,
            has_xdrz: xdrz != null,
            fwhm: isFinite( descriptor.FWHM ) ? descriptor.FWHM : null,
            noise: isFinite( descriptor.noise ) ? descriptor.noise : null,
            signal: isFinite( descriptor.PSFSignalWeight ) ? descriptor.PSFSignalWeight : null,
            exposure: isFinite( frame.exposureTime ) ? frame.exposureTime : null,
            raster: rasterFile,
            background: isFinite( descriptor.median ) ? descriptor.median : null
         } );
      }
      let requestFile = groupDirectory + "/wbpp-request.json";
      File.writeTextFile( requestFile, JSON.stringify( prepareRequest, null, 2 ) );
      cfg.requestFile = requestFile;
      if ( prepareRequest.frames.length == 0 )
      {
         console.warningln( "** Warning: no calibrated frames were exported for DWT-Drizzle group " + groupName + "." );
         continue;
      }
      let result = executeDwtGroup( env, requestInterruption, cfg, requestFile, groupDirectory, groupName );
      if ( result )
         completed++;
      else
      {
         this.statusMessage = "DWT-Drizzle failed for " + groupName;
         return OperationBlockStatus.FAILED;
      }
   }

   if ( completed == 0 )
   {
      console.warningln( "** Warning: no DWT-Drizzle light groups were processed." );
      env.params.processLogger.addWarning( "No DWT-Drizzle light groups were processed." );
      this.hasWarnings = true;
   }
   this.statusMessage = completed + " light group(s) integrated";
   console.noteln( "* DWT-Drizzle handoff complete: " + completed + " group(s)." );
   console.flush();
   return OperationBlockStatus.DONE;
};
engine.operationQueue.operations.splice( dwtOperationIndex, 0, {
   operation: dwtOperationBlock,
   params: {
      processLogger: engine.processLogger
   }
} );
for ( let i = dwtOperationIndex; i < engine.operationQueue.operations.length; ++i )
   engine.operationQueue.operations[ i ].operation.__index__ = i;

function makeDwtRequest( engine, outputDirectory, width, height, config )
{
   return {
      schema: "dwt-wbpp-command/1",
      source: "PixInsight/WBPP",
      calibrated_directory: engine.outputDirectory + "/calibrated",
      registered_directory: engine.outputDirectory + "/registered",
      output_directory: outputDirectory,
      frames: [],
      raster: {
         directory: outputDirectory,
         file_extension: "f32le",
         width: width,
         height: height,
         dtype: "float32-le"
      },
      options: {
         clarity_exponent: 20.0,
         snr_exponent: 2.0,
         snr_clip: 3.0,
         detail_snr: config.detailSnr,
         flux_power: config.fluxPower,
         sharp_weight: 0.0,
         rejection_sigma: 3.5,
         rejection_fraction: 1.0 / 3.0,
         drizzle_scale: config.drizzleScale,
         drizzle_pixfrac: config.drizzlePixfrac,
         allow_missing_xdrz: true,
         env_window_multiplier: 2.0,
         env_epsilon: 16.0,
         env_power: 1.0,
         r_floor: 0.0,
         r_mode: config.rFieldMode,
         r_window: config.rFieldWindow,
         r_contrast: config.rFieldContrast,
         ref_frames: config.referenceFrames
      }
   };
}

function executeDwtGroup( env, requestInterruption, cfg, requestFile, groupDirectory, groupName )
{
   console.noteln( "* DWT-Drizzle: " + groupName );
   if ( !File.exists( cfg.executable ) )
   {
      console.criticalln( "** Error: Rust backend executable not found: " + cfg.executable );
      env.params.processLogger.addError( "Rust backend executable not found: " + cfg.executable );
      requestInterruption();
      return false;
   }
   let resultFile = groupDirectory + "/dwt-result.json";
   let result = runExternalProcess( cfg.executable, [ "wbpp-pws", requestFile, resultFile ], function( progress )
   {
      dwtOperationBlock.groupDescription = groupName + " | " + progress;
      console.noteln( "* DWT-Drizzle " + progress );
      console.flush();
   } );
   if ( !result.ok )
   {
      console.criticalln( "** Error: DWT-Drizzle failed for " + groupName + "." );
      if ( result.stdout.length > 0 )
         console.writeln( result.stdout );
      if ( result.stderr.length > 0 )
         console.criticalln( result.stderr );
      env.params.processLogger.addError( "DWT-Drizzle failed for " + groupName + "." );
      requestInterruption();
      return false;
   }
   if ( result.stdout.length > 0 )
      console.writeln( result.stdout );
   let resultJson;
   try
   {
      resultJson = JSON.parse( File.readTextFile( resultFile ) );
   }
   catch ( error )
   {
      console.criticalln( "** Error: Rust result metadata could not be read for " + groupName + "." );
      env.params.processLogger.addError( "Rust result metadata could not be read for " + groupName + "." );
      requestInterruption();
      return false;
   }
   if ( resultJson.status != "ok" || !Array.isArray( resultJson.raster_shape )
      || resultJson.raster_shape.length != 2
      || !resultJson.raster_shape.every( value => Number.isInteger( value ) && value > 0 ) )
   {
      env.params.processLogger.addError( "Invalid Rust result metadata for " + groupName + "." );
      requestInterruption();
      return false;
   }
   let outputWidth = resultJson.raster_shape[ 1 ];
   let outputHeight = resultJson.raster_shape[ 0 ];
   let frameSummary = "DWT-Drizzle registered " + resultJson.registered_frames + "/" + resultJson.frame_count
      + " frames for " + groupName + ", sharp reference frames: " + resultJson.sharp_reference_frames;
   env.params.processLogger.addMessage( frameSummary );
   for ( let warning of ( resultJson.warnings || [] ) )
   {
      console.warningln( "** Warning: " + warning );
      env.params.processLogger.addWarning( warning );
   }
   let outputFile = String( resultJson.output_raster );
   let outputWindow = readFloat32Window( outputFile, outputWidth, outputHeight );
   if ( outputWindow == null )
   {
      console.criticalln( "** Error: Rust PWS output could not be loaded for " + groupName + "." );
      env.params.processLogger.addError( "Rust PWS output could not be loaded for " + groupName + "." );
      requestInterruption();
      return false;
   }
   outputWindow.mainView.id = "DWT_" + groupName.replace( /[^A-Za-z0-9_]/g, "_" );
   let outputPath = groupDirectory + "/master_dwt_" + groupName + ".xisf";
   dwtOperationBlock.groupDescription = groupName + " | Saving master XISF";
   console.noteln( "* DWT-Drizzle saving master XISF for " + groupName );
   console.flush();
   let saved = outputWindow.saveAs( outputPath, false, false, true, false );
   outputWindow.forceClose();
   if ( !saved )
   {
      env.params.processLogger.addError( "DWT-Drizzle could not save " + outputPath );
      requestInterruption();
      return false;
   }
   console.noteln( "  output: <raw>" + outputPath + "</raw>" );
   env.params.processLogger.addSuccess( "DWT-Drizzle PWS generated", outputPath );
   return true;
}

function readDwtConfiguration( engine )
{
   let scriptPath = engine.pipelineBuilderScriptFile;
   let scriptDirectory = File.extractDrive( scriptPath ) + File.extractDirectory( scriptPath );
   let prefix = scriptDirectory.length > 0 ? scriptDirectory + "/" : "";
   let configFile = prefix + "dwt-backend.json";
   let projectConfigFile = engine.outputDirectory + "/DWT/dwt-backend.json";
   let config = {
      executable: prefix + "bin/dwt-backend.exe",
      requestFile: "",
      useXdrzSeeds: false,
      detailSnr: 0.0,
      fluxPower: 0.0,
      drizzleScale: 1.0,
      drizzlePixfrac: 1.0,
      rFieldMode: "max",
      rFieldWindow: 0.0,
      rFieldContrast: 3.0,
      referenceFrames: 0
   };
   if ( File.exists( configFile ) )
   {
      applyDwtConfiguration( config, configFile );
   }
   if ( File.exists( projectConfigFile ) )
   {
      applyDwtConfiguration( config, projectConfigFile );
   }
   return config;
}

function applyDwtConfiguration( config, configFile )
{
   try
   {
      let parsed = JSON.parse( File.readTextFile( configFile ) );
      if ( parsed.executable )
         config.executable = parsed.executable;
      if ( parsed.requestFile )
         config.requestFile = parsed.requestFile;
      if ( typeof parsed.useXdrzSeeds == "boolean" )
         config.useXdrzSeeds = parsed.useXdrzSeeds;
      if ( typeof parsed.detailSnr == "number" && isFinite( parsed.detailSnr ) )
         config.detailSnr = parsed.detailSnr;
      if ( typeof parsed.fluxPower == "number" && isFinite( parsed.fluxPower ) )
         config.fluxPower = parsed.fluxPower;
      if ( typeof parsed.drizzleScale == "number" && isFinite( parsed.drizzleScale ) && parsed.drizzleScale > 0 )
         config.drizzleScale = parsed.drizzleScale;
      if ( typeof parsed.drizzlePixfrac == "number" && parsed.drizzlePixfrac > 0 && parsed.drizzlePixfrac <= 1 )
         config.drizzlePixfrac = parsed.drizzlePixfrac;
      if ( [ "max", "band", "ctn" ].indexOf( parsed.rFieldMode ) >= 0 )
         config.rFieldMode = parsed.rFieldMode;
      if ( typeof parsed.rFieldWindow == "number" && isFinite( parsed.rFieldWindow ) && parsed.rFieldWindow >= 0 )
         config.rFieldWindow = parsed.rFieldWindow;
      if ( typeof parsed.rFieldContrast == "number" && isFinite( parsed.rFieldContrast ) && parsed.rFieldContrast > 0.8 )
         config.rFieldContrast = parsed.rFieldContrast;
      if ( Number.isInteger( parsed.referenceFrames ) && parsed.referenceFrames >= 0 && parsed.referenceFrames <= 64 )
         config.referenceFrames = parsed.referenceFrames;
   }
   catch ( error )
   {
      console.warningln( "** Warning: invalid DWT configuration: " + error.message );
   }
}

function sanitizeDwtName( value )
{
   return String( value ).replace( /[^A-Za-z0-9_.-]+/g, "_" );
}

function dwtSourceForFrame( frame )
{
   if ( !frame || !frame.fileItem )
      return undefined;
   let processed = frame.fileItem.processed && frame.fileItem.processed[ frame.associatedID ];
   if ( processed )
   {
      if ( processed[ BPP.FrameProcessingStep.DEBAYER ] )
         return processed[ BPP.FrameProcessingStep.DEBAYER ];
      if ( processed[ BPP.FrameProcessingStep.CALIBRATION ] )
         return processed[ BPP.FrameProcessingStep.CALIBRATION ];
   }
   let defaults = frame.fileItem.processed && frame.fileItem.processed[ BPP.Defaults.associatedId ];
   if ( defaults )
   {
      if ( defaults[ BPP.FrameProcessingStep.DEBAYER ] )
         return defaults[ BPP.FrameProcessingStep.DEBAYER ];
      if ( defaults[ BPP.FrameProcessingStep.CALIBRATION ] )
         return defaults[ BPP.FrameProcessingStep.CALIBRATION ];
   }
   return undefined;
}

function findDwtInsertionIndex()
{
   let operations = engine.operationQueue.operations;
   let lastRegistration = -1;
   let firstIntegration = operations.length;
   for ( let i = 0; i < operations.length; ++i )
   {
      let operation = operations[ i ] && operations[ i ].operation;
      if ( !operation || !operation.envForScript )
         continue;
      let info = operation.envForScript();
      if ( !info || !info.group || info.group.imageType != ImageType.Light )
         continue;
      let name = String( info.name || "" );
      if ( name == "Registration" )
         lastRegistration = i;
      if ( name == "Integration" || name == "FastIntegration"
         || name == "Drizzle Integration" )
         firstIntegration = Math.min( firstIntegration, i );
   }
   if ( lastRegistration >= 0 )
      return Math.min( lastRegistration + 1, firstIntegration );
   return firstIntegration;
}

function runExternalProcess( executable, args, onProgress )
{
   let process = new ExternalProcess;
   let stdout = "";
   let stderr = "";
   let started = false;
   let exitCode = null;
   let lastProgress = 0;

   function lineReader( append )
   {
      let pending = "";
      return function( chunk, finished )
      {
         let lines = ( pending + chunk ).split( "\n" );
         pending = lines.pop();
         if ( finished && pending.length > 0 )
         {
            lines.push( pending );
            pending = "";
         }
         for ( let line of lines )
         {
            line = line.replace( /\r$/, "" );
            let progress = /^\[DWT\] (\d+) (.*)$/.exec( line );
            if ( progress != null )
            {
               let sequence = Number( progress[ 1 ] );
               if ( sequence > lastProgress )
               {
                  lastProgress = sequence;
                  onProgress( progress[ 2 ] );
               }
            }
            else
               append( line + "\n" );
         }
      };
   }

   let readStdout = lineReader( line => { stdout += line; } );
   let readStderr = lineReader( line => { stderr += line; } );
   process.onStarted = () => { started = true; };
   process.onFinished = ( code ) => { exitCode = code; };
   process.onStandardOutputDataAvailable = () => { readStdout( String( process.stdout ), false ); };
   process.onStandardErrorDataAvailable = () => { readStderr( String( process.stderr ), false ); };

   try
   {
      if ( process.start( executable, args ) === false )
         return { ok: false, stdout: stdout, stderr: stderr };
   }
   catch ( error )
   {
      return { ok: false, stdout: stdout, stderr: stderr + error.message };
   }

   while ( process.isStarting )
      CoreApplication.processEvents();
   while ( process.isRunning )
      CoreApplication.processEvents();
   readStdout( String( process.stdout ), true );
   readStderr( String( process.stderr ), true );

   return {
      ok: started && exitCode === 0,
      stdout: stdout,
      stderr: stderr
   };
}

function writeCalibratedFloat32( sourcePath, outputPath, width, height )
{
   let window = ImageWindow.open( sourcePath )[ 0 ];
   if ( window == null || window.isNull )
      return false;
   let image = window.mainView.image;
   if ( image.numberOfNominalChannels != 1 || image.width != width || image.height != height )
   {
      window.forceClose();
      return false;
   }
   image.resetSelections();
   image.selectedChannel = 0;
   let samples = new Float32Array( image.width * image.height );
   image.getSamples( samples );
   window.forceClose();
   let file = File.createFileForWriting( outputPath );
   file.write( samples );
   file.close();
   return true;
}

function readFloat32Window( sourcePath, width, height )
{
   if ( !File.exists( sourcePath ) )
      return null;
   let file = File.openFileForReading( sourcePath );
   let samples = file.read( DataType.Float32Array, width * height );
   file.close();
   let window = new ImageWindow( width, height, 1, 32, true, false, "DWT_PWS" );
   window.mainView.beginProcess( UndoFlag.NoSwapFile );
   window.mainView.image.setSamples( samples );
   window.mainView.endProcess();
   window.show();
   return window;
}
