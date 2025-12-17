function GUI
% DIGITAL COMMUNICATIONS PROJECT GUI (Q3 → Q9)
% Single-file implementation

    

    % ================= HOME PAGE =================
    homeFig = uifigure( ...
        'Name','Channel Coding Project', ...
        'Position',[500 200 420 500]);

    gl = uigridlayout(homeFig,[8 1]);
    gl.RowHeight = {'fit','1x','1x','1x','1x','1x','1x','1x'};
    gl.Padding = [20 20 20 20];

    % Title
    uilabel(gl, ...
        'Text','Channel Coding Project', ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Question buttons
    for q = 3:9
        uibutton(gl, ...
            'Text', sprintf('Question %d', q), ...
            'FontSize',14, ...
            'ButtonPushedFcn', @(~,~) open_question(q, homeFig));
    end
end

% ===============================================================
%                  QUESTION WINDOW
% ===============================================================
function open_question(qnum, homeFig)

    qFig = uifigure( ...
        'Name', sprintf('Question %d', qnum), ...
        'Position',[400 150 900 550]);

    gl = uigridlayout(qFig,[3 1]);
    gl.RowHeight = {'fit','fit','1x'};
    gl.Padding = [15 15 15 15];

    % Header
    uilabel(gl, ...
        'Text', sprintf('Question %d', qnum), ...
        'FontSize',16, ...
        'FontWeight','bold', ...
        'HorizontalAlignment','center');

    % Return button
    uibutton(gl, ...
        'Text','⬅ Return to Home', ...
        'FontSize',13, ...
        'ButtonPushedFcn', @(~,~) return_home(qFig, homeFig));

    % Output console (replacement for Command Window)
    logBox = uitextarea(gl, ...
        'Editable','off', ...
        'FontSize',11);

    drawnow;

    % ================= RUN QUESTION WITH OUTPUT CAPTURE =================
    try
        switch qnum
            case 3
                outputText = evalc('Q3');
            case 4
                outputText = evalc('Q4');
            case 5
                outputText = evalc('Q5');
            case 6
                outputText = evalc('Q6');
            case 7
                outputText = evalc('Q7');
            case 8
                outputText = evalc('Q8');
            case 9
                outputText = evalc('Q9');
        end

        logBox.Value = splitlines(outputText);

    catch ME
        logBox.Value = { ...
            '❌ Error occurred:', ...
            ME.message ...
        };
        uialert(qFig, ME.message, 'Execution Error');
    end
end

% ===============================================================
%                  RETURN BUTTON
% ===============================================================
function return_home(qFig, homeFig)
    if isvalid(qFig)
        close(qFig);
    end
    homeFig.Visible = 'on';
end
