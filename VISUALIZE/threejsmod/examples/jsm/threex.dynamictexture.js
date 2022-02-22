var THREEx	= THREEx	|| {}

//////////////////////////////////////////////////////////////////////////////////
//		Constructor							//
//////////////////////////////////////////////////////////////////////////////////

/**
 * create a dynamic texture with a underlying canvas
 * 
 * @param {Number} width  width of the canvas
 * @param {Number} height height of the canvas
 */
THREEx.DynamicTexture = function(width, height){
	var canvas	= document.createElement( 'canvas' )
	canvas.width	= width
	canvas.height	= height
	this.canvas	= canvas

	var context	= canvas.getContext( '2d' )	
	this.context	= context
	
	var texture	= new THREE.Texture(canvas)
	this.texture	= texture
}

//////////////////////////////////////////////////////////////////////////////////
//		methods								//
//////////////////////////////////////////////////////////////////////////////////

/**
 * clear the canvas
 * 
 * @param  {String*} fillStyle 		the fillStyle to clear with, if not provided, fallback on .clearRect
 * @return {THREEx.DynamicTexture}      the object itself, for chained texture
 */
THREEx.DynamicTexture.prototype.clear = function(fillStyle){
	// depends on fillStyle
	if( fillStyle !== undefined ){
		this.context.fillStyle	= fillStyle
		this.context.fillRect(0,0,this.canvas.width, this.canvas.height)		
	}else{
		this.context.clearRect(0,0,this.canvas.width, this.canvas.height)		
	}
	// make the texture as .needsUpdate
	this.texture.needsUpdate	= true;
	// for chained API 
	return this;
}

/**
 * draw text
 * 
 * @param  {String}		text	the text to display
 * @param  {Number|undefined}	x	if provided, it is the x where to draw, if not, the text is centered
 * @param  {Number}		y	the y where to draw the text
 * @param  {String*} 		fillStyle the fillStyle to clear with, if not provided, fallback on .clearRect
 * @param  {String*} 		contextFont the font to use
 * @return {THREEx.DynamicTexture}	the object itself, for chained texture
 */
THREEx.DynamicTexture.prototype.drawText = function(text, dx, dy, fillStyle, contextFont){
	// set font if needed
	if( contextFont !== undefined )	this.context.font = contextFont;
	// if x isnt provided 
	var textSize = this.context.measureText(text);
	x = (this.canvas.width - textSize.width) / 2;
	y = (this.canvas.height/2); // - textSize.height) / 2;
	if (!( dx === undefined || dx === null )){
		x = x+dx;
	}
	if (!( dy === undefined || dy === null )){
		y = y+dy;
	}
	// actually draw the text
	this.context.fillStyle = fillStyle;
	this.context.fillText(text, x, y);
	// make the texture as .needsUpdate
	this.texture.needsUpdate	= true;
	// for chained API 
	return this;
};

THREEx.DynamicTexture.prototype.drawTextCooked = function(text, options){
	var context	= this.context
	var canvas	= this.canvas
	options		= options	|| {}
	var params	= {
		margin		: options.margin !== undefined ? options.margin	: 0.1,
		lineHeight	: options.lineHeight !== undefined ? options.lineHeight : 0.1,
		align		: options.align !== undefined ? options.align : 'left',
		fillStyle	: options.fillStyle !== undefined ? options.fillStyle : 'black',
	}
	context.save()
	context.fillStyle	= params.fillStyle;

	var y	= (params.lineHeight + params.margin)*canvas.height
	while(text.length > 0 ){
		// compute the text for specifically this line
		var maxText	= computeMaxTextLength(text)
		// update the remaining text
		text	= text.substr(maxText.length)


		// compute x based on params.align
		var textSize	= context.measureText(maxText);
		if( params.align === 'left' ){
			var x	= params.margin*canvas.width
		}else if( params.align === 'right' ){
			var x	= (1-params.margin)*canvas.width - textSize.width
		}else if( params.align === 'center' ){
			var x = (canvas.width - textSize.width) / 2;
		}else	console.assert( false )

		// actually draw the text at the proper position
		this.context.fillText(maxText, x, y);

		// goto the next line
		y += params.lineHeight*canvas.height
	}
	context.restore()

	// make the texture as .needsUpdate
	this.texture.needsUpdate	= true;
	// for chained API
	return this;

	function computeMaxTextLength(text){
		var maxText	= ''
		var maxWidth	= (1-params.margin*2)*canvas.width
		while( maxText.length !== text.length ){
			var textSize	= context.measureText(maxText);
			if( textSize.width > maxWidth )	break;
			maxText	+= text.substr(maxText.length, 1)
		}
		return maxText
	}
}

/**
 * execute the drawImage on the internal context
 * the arguments are the same the official context2d.drawImage
 */
THREEx.DynamicTexture.prototype.drawImage = function(/* same params as context2d.drawImage */){
	// call the drawImage
	this.context.drawImage.apply(this.context, arguments)
	// make the texture as .needsUpdate
	this.texture.needsUpdate	= true;
	// for chained API 
	return this;
}



