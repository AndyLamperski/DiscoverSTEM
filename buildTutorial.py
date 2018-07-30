import pweave as pw
import discoverSTEM.utils as ut

filename = 'tutorial.pmd'
tutorial_text = ut.set_pweave_variables(filename,{'text_mode' : True,
                                               'debug_mode' : False},
                                     newname = 'tutorial_notebook.pmd')

debug_text = ut.set_pweave_variables(filename,{'text_mode' : True,
                                            'debug_mode' : True},
                                  newname = 'debug_notebook.pmd')

code_text = ut.set_pweave_variables(filename,{'text_mode' : False,
                                           'debug_mode' : False},
                                 newname = 'checker_code.pmd')


pw.tangle(code_text,informat='noweb',output = 'discoverSTEM/checker.py')
pw.weave(tutorial_text,informat='noweb',doctype='notebook',
         output = 'tutorial_notebook.ipynb')

# pw.weave(debug_text,informat='noweb',doctype='notebook',
#          output = 'debug_notebook.ipynb')
